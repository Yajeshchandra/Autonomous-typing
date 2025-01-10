#!/usr/bin/env python3

# import rospy
# from sensor_msgs.msg import Image
# from cv_bridge import CvBridge
# import cv2
# import numpy as np

# # Initialize global variables
# bridge = CvBridge()
# left_image = None
# right_image = None

# def left_image_callback(msg):
#     """Callback for the left camera."""
#     global left_image
#     left_image = bridge.imgmsg_to_cv2(msg, "bgr8")

# def right_image_callback(msg):
#     """Callback for the right camera."""
#     global right_image
#     right_image = bridge.imgmsg_to_cv2(msg, "bgr8")

# def detect_cylinder(image):
#     """Detect a cylinder in the given image."""
#     # Convert to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # Use GaussianBlur to reduce noise
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)

#     # Edge detection using Canny
#     edges = cv2.Canny(blurred, 50, 150)

#     # Find contours
#     contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     for contour in contours:
#         # Fit an ellipse if the contour has enough points
#         if len(contour) >= 5:
#             ellipse = cv2.fitEllipse(contour)
#             (x, y), (major_axis, minor_axis), angle = ellipse

#             # Ensure dimensions are valid
#             if major_axis > 0 and minor_axis > 0:
#                 # Check aspect ratio to identify a cylinder-like shape
#                 aspect_ratio = max(major_axis, minor_axis) / min(major_axis, minor_axis)
#                 if 1.2 < aspect_ratio < 8:  # Adjust thresholds as needed
#                     return int(x), int(y), int(major_axis), int(minor_axis)
#     return None


# def compute_depth(x_left, x_right, focal_length, baseline):
#     """Compute the depth (Z) using disparity."""
#     disparity = x_left - x_right
#     if disparity > 0:
#         depth = (focal_length * baseline) / disparity
#         return depth
#     return None

# def compute_coordinates(x_left, y_left, depth, cx, cy, focal_length):
#     """Compute the real-world coordinates (X, Y, Z) in the world frame."""
#     # Coordinates relative to the left camera
#     X_cam = (x_left - cx) * depth / focal_length
#     Y_cam = (y_left - cy) * depth / focal_length
#     Z_cam = depth

#     # Transform to world coordinates
#     X_world = X_cam - 0.05  # Adjust for left camera's X position
#     Y_world = -Y_cam         # Flip Y-axis to get positive values upwards
#     Z_world = Z_cam         # Z remains the same

#     return X_world, Y_world, Z_world

# def main():
#     rospy.init_node("stereo_vision_cylinder_detection")

#     # Subscribers for stereo images
#     rospy.Subscriber("/camera_base/image_raw", Image, left_image_callback)
#     rospy.Subscriber("/camera_base_2/image_raw", Image, right_image_callback)

#     # Camera intrinsic parameters (example values for 1920x1080 resolution)
#     cx = 960  # Principal point x-coordinate
#     cy = 540  # Principal point y-coordinate
#     focal_length = 520  # Focal length in pixels (adjust based on your camera)
#     baseline = 0.1  # Distance between cameras in meters

#     rate = rospy.Rate(10)  # 10 Hz
#     while not rospy.is_shutdown():
#         if left_image is not None and right_image is not None:
#             # Detect cylinder in both images
#             left_result = detect_cylinder(left_image)
#             right_result = detect_cylinder(right_image)

#             if left_result and right_result:
#                 x_left, y_left, major_left, minor_left = left_result
#                 x_right, y_right, major_right, minor_right = right_result

#                 # Compute depth (Z)
#                 depth = compute_depth(x_left, x_right, focal_length, baseline)

#                 if depth:
#                     # Compute real-world coordinates (X, Y, Z)
#                     X, Y, Z = compute_coordinates(x_left, y_left, depth, cx, cy, focal_length)

#                     label = f"X: {X:.2f} m, Y: {Y:.2f} m, Z: {Z:.2f} m"
                    
#                     rospy.loginfo(label)

#                     # Draw bounding ellipse and label on the left image
#                     cv2.ellipse(left_image, ((x_left, y_left), (major_left, minor_left), 0), (0, 255, 0), 2)
#                     cv2.putText(left_image, label, (x_left - 50, y_left - 10), 
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

#             # Show the left image with annotations
#             cv2.imshow("Left Camera - Cylinder Detection", left_image)

#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#         rate.sleep()

#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()
















import tf2_ros
from geometry_msgs.msg import PointStamped
from tf2_geometry_msgs import do_transform_point

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

# Initialize global variables
bridge = CvBridge()
left_image = None
right_image = None


def transform_point(camera_point, camera_frame="camera_link1"):
    # Initialize TF2 buffer and listener
    tf_buffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tf_buffer)

    try:
        # Wait for the transform to be available
        transform = tf_buffer.lookup_transform("world", camera_frame, rospy.Time(0), rospy.Duration(1.0))

        # Transform the point
        world_point = do_transform_point(camera_point, transform)
        return world_point

    except tf2_ros.LookupException as e:
        rospy.logerr(f"Transform lookup failed: {e}")
        return None

def left_image_callback(msg):
    """Callback for the left camera."""
    global left_image
    left_image = bridge.imgmsg_to_cv2(msg, "bgr8")

def right_image_callback(msg):
    """Callback for the right camera."""
    global right_image
    right_image = bridge.imgmsg_to_cv2(msg, "bgr8")

def detect_red_object(image):
    """Detect a red object in the given image."""
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define red color range
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    # Create masks for both red ranges
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        # Calculate the bounding rectangle for each contour
        x, y, w, h = cv2.boundingRect(contour)

        # Set a minimum size to filter noise
        if w > 20 and h > 20:  # Adjust these thresholds based on your needs
            return x, y, w, h
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
    
    

    # return X_world, Y_world, Z_world
    
    return X_cam, Y_cam, Z_cam

def main():
    rospy.init_node("stereo_vision_red_object_detection")

    # Subscribers for stereo images
    rospy.Subscriber("/camera_base/image_raw", Image, left_image_callback)
    rospy.Subscriber("/camera_base_2/image_raw", Image, right_image_callback)

    # Camera intrinsic parameters (example values for 800*800 resolution)
    
    cx = 450  # Principal point x-coordinate
    cy = 400  # Principal point y-coordinate
    focal_length = 610  # Focal length in pixels (adjust based on your camera)
    baseline = 0.24  # Distance between cameras in meters

    rate = rospy.Rate(10)  # 10 Hz
    while not rospy.is_shutdown():
        if left_image is not None and right_image is not None:
            # Detect red object in both images
            left_result = detect_red_object(left_image)
            right_result = detect_red_object(right_image)

            if left_result and right_result:
                x_left, y_left, w_left, h_left = left_result
                x_right, y_right, w_right, h_right = right_result

                # Compute depth (Z)
                depth = compute_depth(x_left, x_right, focal_length, baseline)

                if depth:
                    # Compute real-world coordinates (X, Y, Z)
                    X_camera, Y_camera, Z_camera = compute_coordinates(x_left + w_left // 2, y_left + h_left // 2, depth, cx, cy, focal_length)
                    
                  
                    label = f"X: {X_camera:.2f} m, Y: {Y_camera:.2f} m, Z: {Z_camera:.2f} m"
                    
                    rospy.loginfo(f"Object in camera frame: x: {X_camera}, y: {Y_camera}, z: {Z_camera}")

                    
                    camera_point = PointStamped()
                    camera_point.header.frame_id = "camera_base_left"
                    camera_point.point.x = X_camera  # Replace with actual coordinates
                    camera_point.point.y = Y_camera
                    camera_point.point.z = Z_camera
                    
                    # X_world = X_cam  # Adjust for left camera's X position
                    # Y_world = -Y_cam         # Flip Y-axis to get positive values upwards
                    # Z_world = Z_cam         # Z remains the same
                    link = "camera_link1"
                    world_point = transform_point(camera_point,link)
                    if world_point:
                        rospy.loginfo(f"Object in world frame: x: {world_point.point.x}, y: {world_point.point.y}, z: {world_point.point.z}")
                        
                    rospy.loginfo("")

                    # label = f"X: {X:.2f} m, Y: {Y:.2f} m, Z: {Z:.2f} m"
                    # remember that z is depth, y is positive up, x is positive right
                    
                    # rospy.loginfo(label)

                    # Draw bounding box and label on the left image
                    cv2.rectangle(left_image, (x_left, y_left), (x_left + w_left, y_left + h_left), (0, 255, 0), 2)
                   
                    # cv2.putText(left_image, label, (x_left, y_left - 10), 
                                # cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Show the left image with annotations
            cv2.imshow("Left Camera - Red Object Detection", left_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        rate.sleep()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
