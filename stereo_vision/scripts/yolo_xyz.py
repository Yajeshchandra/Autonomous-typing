#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import torch

# Initialize global variables
bridge = CvBridge()
left_image = None
right_image = None
model = None  # YOLO model

def left_image_callback(msg):
    """Callback for the left camera."""
    global left_image
    left_image = bridge.imgmsg_to_cv2(msg, "bgr8")

def right_image_callback(msg):
    """Callback for the right camera."""
    global right_image
    right_image = bridge.imgmsg_to_cv2(msg, "bgr8")

def detect_object_yolo(image):
    """Detect objects using YOLO."""
    global model
    results = model(image)  # Inference using YOLO
    objects = []
    
    for result in results.xyxy[0]:  # Each detection
        x1, y1, x2, y2, conf, cls = map(int, result[:6])
        label = model.names[cls]  # Get label from class ID
        x_center = (x1 + x2) // 2
        y_center = (y1 + y2) // 2
        objects.append((label, x_center, y_center, x1, y1, x2, y2, conf))
        
    return objects

def compute_depth(x_left, x_right, focal_length, baseline):
    """Compute the depth (Z) using disparity."""
    disparity = x_left - x_right
    if disparity > 0:
        return (focal_length * baseline) / disparity
    return None

def compute_coordinates(x_left, y_left, depth, cx, cy, focal_length):
    """Compute the real-world coordinates (X, Y, Z) in the world frame."""
    # Coordinates relative to the left camera
    X_cam = (x_left - cx) * depth / focal_length
    Y_cam = (y_left - cy) * depth / focal_length
    Z_cam = depth

    # Transform to world coordinates
    X_world = X_cam - 0.05  # Adjust for left camera's X position
    Y_world = -Y_cam         # Flip Y-axis to get positive values upwards
    Z_world = Z_cam          # Z remains the same

    return X_world, Y_world, Z_world

def main():
    global model

    rospy.init_node("stereo_vision_yolo_depth")

    # Load YOLO model (YOLOv5 from PyTorch Hub)
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    # Subscribers for stereo images
    rospy.Subscriber("/camera/left/color/image_raw", Image, left_image_callback)
    rospy.Subscriber("/camera/right/color/image_raw", Image, right_image_callback)

    # Camera intrinsic parameters (example values for 1920x1080 resolution)
    cx = 960  # Principal point x-coordinate
    cy = 540  # Principal point y-coordinate
    focal_length = 520  # Focal length in pixels (adjust based on your camera)
    baseline = 0.1  # Distance between cameras in meters

    rate = rospy.Rate(10)  # 10 Hz
    print("Running...")
    while not rospy.is_shutdown():
        if left_image is not None and right_image is not None:
            # Detect objects in both images using YOLO
            left_objects = detect_object_yolo(left_image)
            right_objects = detect_object_yolo(right_image)
            
            print(f"Left: {len(left_objects)}, Right: {len(right_objects)}")
            

            display_image = np.hstack((left_image, right_image))  # Combine images horizontally
            # change the cv.rectangle parameter to use this format

            for left_obj in left_objects:
                label_left, x_left, y_left, *_ = left_obj

                # Find corresponding object in the right image
                for right_obj in right_objects:
                    label_right, x_right, y_right, *_ = right_obj

                    if label_left == label_right:  # Match based on label
                        # Compute depth (Z)
                        depth = compute_depth(x_left, x_right, focal_length, baseline)
                        
                        if depth:
                            print(f"Depth: {depth:.2f} m")
                        else:
                            print("No depth")

                        if depth:
                            # Compute real-world coordinates (X, Y, Z)
                            X, Y, Z = compute_coordinates(x_left, y_left, depth, cx, cy, focal_length)

                            label = f"{label_left} - X: {X:.2f} m, Y: {Y:.2f} m, Z: {Z:.2f} m"

                            # Draw bounding box and label on the left image
                            cv2.rectangle(left_image, (left_obj[3], left_obj[4]), 
                                          (left_obj[5], left_obj[6]), (0, 0, 255), 2)
                            cv2.putText(left_image, label, (left_obj[3], left_obj[4] - 10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Show the combined image with annotations
            cv2.imshow("Stereo Vision - YOLO Depth Estimation", left_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        rate.sleep()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
