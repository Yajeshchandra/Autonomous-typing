#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

# Initialize global variables
bridge = CvBridge()
left_image = None
right_image = None

def left_image_callback(msg):
    """Callback for the left camera."""
    global left_image
    left_image = bridge.imgmsg_to_cv2(msg, "bgr8")

def right_image_callback(msg):
    """Callback for the right camera."""
    global right_image
    right_image = bridge.imgmsg_to_cv2(msg, "bgr8")

def detect_red_sphere(image):
    """Detect a red sphere in the given image."""
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define red color range
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    # Threshold the HSV image to get red colors
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        # Fit a circle around the detected contour
        ((x, y), radius) = cv2.minEnclosingCircle(contour)
        if radius > 1:  # Minimum size threshold
            return int(x), int(y), int(radius)
    return None

def compute_depth(x_left, x_right, focal_length, baseline):
    """Compute the depth (Z) using disparity."""
    disparity = x_left - x_right
    if disparity > 0:
        depth = (focal_length * baseline) / disparity
        return depth
    return None

def compute_coordinates(x_left, y_left, depth, cx, cy, focal_length):
    """Compute the real-world coordinates (X, Y, Z) in the world frame."""
    # Coordinates relative to the left camera
    X_cam = (x_left - cx) * depth / focal_length
    Y_cam = (y_left - cy) * depth / focal_length
    Z_cam = depth

    # Transform to world coordinates
    X_world = X_cam - 0.05  # Adjust for left camera's X position
    Y_world = -Y_cam         # flip Y-axis to get positive values upwards
    Z_world = Z_cam         # Z remains the same

    return X_world, Y_world, Z_world

def main():
    rospy.init_node("stereo_vision_red_sphere")

    # Subscribers for stereo images
    rospy.Subscriber("/camera/left/color/image_raw", Image, left_image_callback)
    rospy.Subscriber("/camera/right/color/image_raw", Image, right_image_callback)

    # Camera intrinsic parameters (example values for 1920x1080 resolution)
    cx = 960  # Principal point x-coordinate
    cy = 540  # Principal point y-coordinate
    focal_length = 520  # Focal length in pixels (adjust based on your camera)
    baseline = 0.1  # Distance between cameras in meters

    rate = rospy.Rate(10)  # 10 Hz
    while not rospy.is_shutdown():
        if left_image is not None and right_image is not None:
            # Detect red sphere in both images
            left_result = detect_red_sphere(left_image)
            right_result = detect_red_sphere(right_image)

            if left_result and right_result:
                x_left, y_left, radius_left = left_result
                x_right, y_right, radius_right = right_result
                

                # Compute depth (Z)
                depth = compute_depth(x_left, x_right, focal_length, baseline)

                if depth:
                    # Compute real-world coordinates (X, Y, Z)
                    X, Y, Z = compute_coordinates(x_left, y_left, depth, cx, cy, focal_length)

                    label = f"X: {X:.2f} m, Y: {Y:.2f} m, Z: {Z:.2f} m"

                    # Draw bounding box and label on the left image
                    cv2.rectangle(left_image, (x_left - radius_left, y_left - radius_left), 
                                  (x_left + radius_left, y_left + radius_left), (0, 255, 0), 2)
                    cv2.putText(left_image, label, (x_left - radius_left, y_left - radius_left - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Show the left image with annotations
            cv2.imshow("Left Camera - Red Sphere Detection", left_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        rate.sleep()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
