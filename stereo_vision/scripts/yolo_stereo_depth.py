#!/usr/bin/env python3




import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import torch  # For YOLO (PyTorch-based)

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
    """Compute the depth using disparity."""
    disparity = x_left - x_right
    print(f"Disparity: {disparity}")
    if disparity > 0:
        depth = (focal_length * baseline) / disparity
        return depth
    return None


def compute_3d_coordinates(x_left, x_right, y_left, focal_length, baseline_left, baseline_right):
    """Compute the 3D coordinates (x, y, z) using disparity and camera positions."""
    disparity = x_left - x_right
    if disparity > 0:
        # Depth in meters
        z = (focal_length * baseline_left) / disparity
        
        # Adjust the x and y in the 3D space based on camera positions
        x = (x_left - focal_length) * z / focal_length + baseline_left
        y = (y_left - focal_length) * z / focal_length
        return x, y, z
    return None, None, None

def main():
    global model
    
    rospy.init_node("stereo_vision_yolo_depth")
    
    # Load YOLO model (YOLOv5 from PyTorch Hub)
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    
    # Subscribers for stereo images
    rospy.Subscriber("/camera/left/color/image_raw", Image, left_image_callback)
    rospy.Subscriber("/camera/right/color/image_raw", Image, right_image_callback)
    
    focal_length = 520  # in pixels
    baseline = 0.1      # distance between cameras in meters
    
    rate = rospy.Rate(10)  # 10 Hz
    
    print("Running... now")
    while not rospy.is_shutdown():
        if left_image is not None and right_image is not None:
            # Detect objects in both images using YOLO
            left_objects = detect_object_yolo(left_image)
            right_objects = detect_object_yolo(right_image)
            
            print(f"Left: {len(left_objects)}, Right: {len(right_objects)}")
            
            display_image = np.hstack((left_image, right_image))  # Combine images horizontally
            
            for left_obj, right_obj in zip(left_objects, right_objects):
                label_left, x_left, y_left, *_ = left_obj
                label_right, x_right, y_right, *_ = right_obj
                
                # Compute depth
                depth = compute_depth(x_left, x_right, focal_length, baseline)
                if depth:
                    print(f"Depth: {depth:.2f} m")
                else:
                    print("No depth")
                
                if depth:
                    label = f"{label_left} - Depth: {depth:.2f} m"
                    
                    # Draw bounding box and label on the combined image
                    cv2.rectangle(display_image, (left_obj[3], left_obj[4]), (left_obj[5], left_obj[6]), (0, 255, 0), 2)
                    cv2.putText(display_image, label, (left_obj[3], left_obj[4] - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Show the combined image
            cv2.imshow("Stereo Vision - YOLO Depth Estimation", display_image)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        rate.sleep()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
