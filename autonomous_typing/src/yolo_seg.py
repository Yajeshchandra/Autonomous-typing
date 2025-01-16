#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
import json

class YoloPixelSegmentationNode:
    def __init__(self):
        rospy.init_node('yolo_pixel_segmentation_node', anonymous=True)
        
        # Parameters
        self.input_topic = rospy.get_param('~input_topic', '/camera_gripper/image_raw')
        self.camera_info_topic = rospy.get_param('~camera_info_topic', '/camera_gripper/camera_info')
        self.output_topic = rospy.get_param('~output_topic', '/camera_gripper/processed_image')
        self.class_to_detect = rospy.get_param('~class_to_detect', 66)
        
        # Real-world keyboard dimensions (mm)
        self.KEYBOARD_LENGTH = 354.076
        self.KEYBOARD_WIDTH = 123.444
        
        # Initialize YOLO model
        self.model = YOLO('yolov8n-seg.pt')
        
        # Camera parameters (will be updated from camera_info)
        self.camera_matrix = None
        self.dist_coeffs = None
        
        # Setup ROS communication
        self.bridge = CvBridge()
        self.subscriber = rospy.Subscriber(self.input_topic, Image, self.image_callback)
        self.camera_info_sub = rospy.Subscriber(self.camera_info_topic, CameraInfo, self.camera_info_callback)
        self.publisher = rospy.Publisher(self.output_topic, Image, queue_size=10)
        
        # Load keyboard layout
        with open('/home/ub20/rsws/src/autonomous_typing/src/keyboard_layout.json', 'r') as f:
            self.keyboard_points_dict = json.load(f)
        
        rospy.loginfo("YoloPixelSegmentationNode initialized.")

    def camera_info_callback(self, msg):
        self.camera_matrix = np.array(msg.K).reshape(3, 3)
        self.dist_coeffs = np.array(msg.D)

    def draw_axes(self, img, rvec, tvec, camera_matrix, dist_coeffs, length=50):
        """Draw 3D coordinate axes on the image."""
        # Define the 3D points for coordinate axes
        origin = np.float32([[0, 0, 0]])
        x_axis = np.float32([[length, 0, 0]])
        y_axis = np.float32([[0, length, 0]])
        z_axis = np.float32([[0, 0, length]])
        
        # Project 3D points to image plane
        origin_2d, _ = cv2.projectPoints(origin, rvec, tvec, camera_matrix, dist_coeffs)
        x_2d, _ = cv2.projectPoints(x_axis, rvec, tvec, camera_matrix, dist_coeffs)
        y_2d, _ = cv2.projectPoints(y_axis, rvec, tvec, camera_matrix, dist_coeffs)
        z_2d, _ = cv2.projectPoints(z_axis, rvec, tvec, camera_matrix, dist_coeffs)
        
        # Convert to integer coordinates
        origin_pt = tuple(map(int, origin_2d[0].ravel()))
        x_pt = tuple(map(int, x_2d[0].ravel()))
        y_pt = tuple(map(int, y_2d[0].ravel()))
        z_pt = tuple(map(int, z_2d[0].ravel()))
        
        # Draw the axes
        cv2.line(img, origin_pt, x_pt, (255, 0, 255), 2)  # X-axis in Red
        cv2.line(img, origin_pt, y_pt, (0, 0, 255), 2)  # Y-axis in Green
        cv2.line(img, origin_pt, z_pt, (255, 100, 0), 2)  # Z-axis in Blue
        
        # Add axis labels
        cv2.putText(img, 'X', x_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(img, 'Y', y_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(img, 'Z', z_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Draw origin point
        cv2.circle(img, origin_pt, 3, (255, 255, 255), -1)
        cv2.putText(img, 'O', (origin_pt[0]-10, origin_pt[1]-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    def process_image(self, cv_image, results):
        # Create visualization image
        vis_image = cv_image.copy()
        
        if len(results) == 0:
            return vis_image

        # Get predictions from the first result
        result = results[0]
        
        if len(result.boxes) == 0:
            return vis_image

        # Process each detection
        for idx in range(len(result.boxes)):
            # Get class id
            class_id = int(result.boxes.cls[idx].item())
            
            if class_id == self.class_to_detect:
                # Get bounding box
                box = result.boxes.xyxy[idx].cpu().numpy()
                x1, y1, x2, y2 = map(int, box)
                
                # Get segmentation mask
                if result.masks is not None and idx < len(result.masks):
                    mask = result.masks[idx].data[0].cpu().numpy()
                    mask = (mask > 0.5).astype(np.uint8) * 255
                    mask = cv2.resize(mask, (cv_image.shape[1], cv_image.shape[0]))
                    
                    # Get mask contours
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    # Draw mask outline
                    cv2.drawContours(vis_image, contours, -1, (0, 255, 255), 2)
                
                # Draw bounding box
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Scale and position keyboard points
                keyboard_points = np.array(list(self.keyboard_points_dict.values()))
                box_length = x2 - x1
                box_width = y2 - y1
                scaled_points = (keyboard_points / np.array([self.KEYBOARD_LENGTH, self.KEYBOARD_WIDTH])) * \
                              np.array([box_length, box_width]) + np.array([x1, y1])
                
                if self.camera_matrix is not None:
                    try:
                        # Create 3D model points (assuming keyboard is flat)
                        model_points = np.zeros((len(keyboard_points), 3))
                        model_points[:, 0] = keyboard_points[:, 0]
                        model_points[:, 1] = keyboard_points[:, 1]
                        
                        # Convert points to appropriate format
                        model_points = model_points.astype(np.float32)
                        scaled_points = scaled_points.astype(np.float32)
                        
                        # Solve PnP
                        success, rvec, tvec = cv2.solvePnP(model_points, 
                                                         scaled_points, 
                                                         self.camera_matrix, 
                                                         self.dist_coeffs,
                                                         flags=cv2.SOLVEPNP_ITERATIVE)
                        
                        if success:
                            # Draw coordinate axes
                            self.draw_axes(vis_image, rvec, tvec, self.camera_matrix, self.dist_coeffs)
                            
                            # Project points to get depth
                            projected_points, _ = cv2.projectPoints(model_points, 
                                                                  rvec, 
                                                                  tvec, 
                                                                  self.camera_matrix, 
                                                                  self.dist_coeffs)
                            
                            # Draw points with depth information
                            for i, (point, proj_point) in enumerate(zip(scaled_points, projected_points)):
                                point = point.astype(int)
                                depth = float(tvec[2] + model_points[i, 2])  # Z component
                                
                                # Color gradient based on depth (red=closer, blue=farther)
                                color = (int(255 * (1 - depth/1000)), 0, int(255 * depth/1000))
                                cv2.circle(vis_image, tuple(point), 5, color, -1)
                                
                                # Display depth value
                                cv2.putText(vis_image, 
                                          f'{depth:.1f}', 
                                          (point[0] + 5, point[1] - 5),
                                          cv2.FONT_HERSHEY_SIMPLEX, 
                                          0.3, 
                                          color, 
                                          1)
                    except Exception as e:
                        rospy.logwarn(f"PnP estimation failed: {e}")
                
        return vis_image

    def image_callback(self, msg):
        try:
            # Convert ROS Image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            cv_image = cv2.resize(cv_image, (640, 640))
            
            # Run YOLO inference
            results = self.model(cv_image)
            
            # Process image and add visualizations
            processed_image = self.process_image(cv_image, results)
            
            # Publish processed image
            ros_image = self.bridge.cv2_to_imgmsg(processed_image, encoding="bgr8")
            self.publisher.publish(ros_image)
            
        except Exception as e:
            rospy.logerr(f"Error in image_callback: {e}")

if __name__ == '__main__':
    try:
        node = YoloPixelSegmentationNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
