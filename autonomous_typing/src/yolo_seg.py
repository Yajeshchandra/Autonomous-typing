#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
import json

class YoloPixelSegmentationNode:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('yolo_pixel_segmentation_node', anonymous=True)

        # Parameters
        self.input_topic = rospy.get_param('~input_topic', '/camera_gripper/image_raw')
        self.output_topic = rospy.get_param('~output_topic', '/camera_gripper/segmented_image')
        self.output_topic_bin = rospy.get_param('~output_topic_bin', '/camera_gripper/bin_segmented_image')
        self.class_to_detect = rospy.get_param('~class_to_detect', 66)

        # YOLO model
        self.model = YOLO('yolov8n-seg.pt')  # Replace with the appropriate YOLOv8 segmentation model

        # ROS Topics
        self.bridge = CvBridge()
        self.subscriber = rospy.Subscriber(self.input_topic, Image, self.image_callback)
        self.publisher = rospy.Publisher(self.output_topic, Image, queue_size=10)
        self.publisher_bin = rospy.Publisher(self.output_topic_bin, Image, queue_size=10)

        rospy.loginfo("YoloPixelSegmentationNode initialized.")
    def image_preprocessing(self,image):
            
        resized = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        # Apply a large Gaussian blur to estimate the background
        blur = cv2.GaussianBlur(gray, (51, 51), 0)
        # Subtract the background
        background_subtracted = cv2.subtract(gray, blur)
        
        normalized = cv2.normalize(background_subtracted, None, 0, 255, cv2.NORM_MINMAX)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_result = clahe.apply(normalized)

        # Estimate the illumination
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
        illumination = cv2.morphologyEx(clahe_result, cv2.MORPH_CLOSE, kernel)
        # Correct the image by dividing by the illumination
        illumination_corrected = cv2.divide(clahe_result, illumination, scale=255)

        _, binarized = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binarized

        binarized = cv2.GaussianBlur(binarized, (5, 5), 0)
        kernel = np.ones((5,5), np.uint8)   # 5x5 kernel
        opening = cv2.morphologyEx(binarized, cv2.MORPH_OPEN, kernel)
        binarized = cv2.dilate(binarized, kernel, iterations=3)
        return binarized
    
    def image_callback(self, msg):
        try:
            # Convert ROS Image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            cv_image = cv2.resize(cv_image, (640, 640))

            # YOLOv8 inference
            results = self.model(cv_image)

            # Process results to extract pixel-level segmentation
            segmented_image = self.get_pixel_segmentation(results, cv_image)
            
            # Binarize the segmented image
            binarized_image = self.image_preprocessing(segmented_image)

            # Convert OpenCV image back to ROS Image and publish
            ros_image = self.bridge.cv2_to_imgmsg(segmented_image, encoding="bgr8")
            self.publisher.publish(ros_image)
            ros_image_bin = self.bridge.cv2_to_imgmsg(binarized_image, encoding="mono8")
            self.publisher_bin.publish(ros_image_bin)
        except Exception as e:
            rospy.logerr(f"Error in image_callback: {e}")

    def get_pixel_segmentation(self, results, original_image):
        # Create a blank mask with the same dimensions as the original image
        mask = np.zeros(original_image.shape[:2], dtype=np.uint8)
        x1=0
        y1=0
        x2=0
        y2=0
        # Iterate through all detections in the results
        for box, cls, seg_mask in zip(results[0].boxes.data, results[0].boxes.cls, results[0].masks.data):
            # Extract bounding box coordinates and draw rectangle
            class_id = int(cls.item())  # Class ID of the detection
            if class_id == self.class_to_detect:
                x1, y1, x2, y2 = box[:4].tolist()
                # Extract the segmentation mask for the detected class
                seg = (seg_mask.cpu().numpy() > 0.5).astype(np.uint8) * 255
                mask = cv2.bitwise_or(mask, seg)

        # Create an RGB image with the mask overlaid
        segmented_image = cv2.bitwise_and(original_image, original_image, mask=mask)
        cv2.rectangle(segmented_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        # Load the keyboard layout points from JSON
        with open('/home/ub20/rsws/src/autonomous_typing/src/keyboard_layout.json', 'r') as f:
            keyboard_points_dict = json.load(f)
        keyboard_points = np.array(list(keyboard_points_dict.values()))
        rospy.loginfo(keyboard_points)
        TOTAL_LENGTH_MM = 354.076  # Length in mm
        TOTAL_WIDTH_MM = 123.444   # Width in mm

        box_length = int(x2) - int(x1)
        box_width = int(y2) - int(y1)

        scaled_keyboard_points = (keyboard_points / np.array([TOTAL_LENGTH_MM, TOTAL_WIDTH_MM])) * np.array([box_length, box_width])
        # Add offset of top left corner to keyboard points
        keyboard_points = scaled_keyboard_points + np.array([int(x1), int(y1)]) 
        for point in keyboard_points:
            cv2.circle(segmented_image, tuple(point.astype(int)), radius=5, color=(255, 0, 0), thickness=-1)
        # keyboard_points_dict = np.load('/home/ub20/rsws/src/autonomous_typing/src/keyboard_layout.npy', allow_pickle=True)
        # keyboard_points = keyboard_points_dict.values()
        # keyboard_points = np.array(keyboard_points)
        # rospy.INFO(keyboard_points)
        # TOTAL_LENGTH_MM = 354.076  # Length in mm
        # TOTAL_WIDTH_MM = 123.444   # Width in mm

        # box_length = int(x2) - int(x1)
        # box_width = int(y2) - int(y1)

        # scaled_keyboard_points = (keyboard_points / np.array([TOTAL_LENGTH_MM, TOTAL_WIDTH_MM])) * np.array([box_length, box_width])
        # # keyboard_points = scaled_keyboard_points.astype(int)
        # # Add offset of top left corner to keyboard points
        # keyboard_points = scaled_keyboard_points + np.array([int(x1), int(y1)]) 
        # for point in keyboard_points:
        #     cv2.circle(segmented_image, tuple(point), radius=5, color=(255, 0, 0), thickness=-1)

        return segmented_image


if __name__ == '__main__':
    try:
        node = YoloPixelSegmentationNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
