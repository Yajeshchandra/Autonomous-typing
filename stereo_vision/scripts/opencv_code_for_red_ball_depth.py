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
    
    
    # for debugging
    # cv2.imshow("Red Mask", mask)
    # cv2.waitKey(1)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        # Fit a circle around the detected contour
        ((x, y), radius) = cv2.minEnclosingCircle(contour)
        if radius > 10:  # Minimum size threshold
            return int(x), int(y), int(radius)
    return None

def compute_depth(x_left, x_right, focal_length, baseline):
    """Compute the depth using disparity."""
    disparity = x_left - x_right
    
    print(f"Disparity: {disparity}")
    # disparity = abs(disparity)
    if disparity > 0:
        depth = (focal_length * baseline) / disparity
        return depth
    
    
    return None

def main():
    rospy.init_node("stereo_vision_red_sphere")
    
    # Subscribers for stereo images
    rospy.Subscriber("/camera/left/color/image_raw", Image, left_image_callback)
    rospy.Subscriber("/camera/right/color/image_raw", Image, right_image_callback)
    
    
    focal_length = 520  # in pixels 
    baseline = 0.1      # distance between cameras in meters 
    
    rate = rospy.Rate(10)  # 10 Hz
    while not rospy.is_shutdown():
        if left_image is not None and right_image is not None:
            # Detect red sphere in both images
            left_result = detect_red_sphere(left_image)
            right_result = detect_red_sphere(right_image)
            
            display_image = np.hstack((left_image, right_image))  # Combine images horizontally
            

            if left_result and right_result:
                x_left, y_left, radius_left = left_result
                x_right, y_right, radius_right = right_result
                
                print(f"Left: ({x_left}, {y_left}), Right: ({x_right}, {y_right})")
                
                # Compute depth
                depth = compute_depth(x_left, x_right, focal_length, baseline)
                # print(f"Depth: {depth:.2f} m")
                if depth:
                    label = f"Depth: {depth:.2f} m"
                    
                    # Draw bounding boxes and label on the combined image
                    cv2.rectangle(display_image, (x_left - radius_left, y_left - radius_left), 
                                  (x_left + radius_left, y_left + radius_left), (0, 255, 0), 2)
                    cv2.putText(display_image, label, (x_left - radius_left, y_left - radius_left - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Show the combined image in a single window
            cv2.imshow("Stereo Vision - Red Sphere Detection", display_image)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        rate.sleep()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
