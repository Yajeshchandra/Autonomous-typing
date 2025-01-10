#!/usr/bin/env python3
# Ctrl+f and type TODO to see where to do changes

import pinocchio as pin
import numpy as np
# from sklearn.linear_model import LinearRegression
import rospy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout
from PyQt5.QtCore import Qt, QTimer
from quadprog import solve_qp
import os
from pinocchio.visualize import MeshcatVisualizer
from typing import Dict, List, Tuple
# from pynput.keyboard import Key, KeyCode, Listener
from ultralytics import YOLO
import cv2

TOTAL_LENGTH_MM = 354.076  # Length in mm
TOTAL_WIDTH_MM = 123.444   # Width in mm

class KeyboardTyperWithIK(QWidget):
    def __init__(self, camera_matrix, dist_coeffs):
        super().__init__()
        
        # Initialize ROS node
        rospy.init_node('keyboard_typer_ik')
        self.pub = rospy.Publisher('/body_controller/command', JointTrajectory, queue_size=10)
        
        self.setup_robot()
        self.setup_ui()
        self.setup_control()
        
        # Keyboard typer specific attributes
        self.key_positions: Dict[str, np.ndarray] = {}
        self.home_position: np.ndarray = None
        self.hover_distance = 0.01  # 1cm hover distance
        self.positions_file = "keyboard_positions.npy"
        self.all_keys = [
            "ESC", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10", "F11", "F12",
            "PRTSC", "SCRLK", "PAUSE", "`", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0",
            "-", "=", "BACKSPACE", "INS", "HOME", "PAGEUP", "TAB", "Q", "W", "E", "R", "T", "Y",
            "U", "I", "O", "P", "[", "]", "\\", "DEL", "END", "PAGEDOWN", "CAPSLOCK", "A", "S",
            "D", "F", "G", "H", "J", "K", "L", ";", "'", "ENTER", "SHIFT", "Z", "X", "C", "V",
            "B", "N", "M", ",", ".", "/", "UP", "CTRL", "WIN", "ALT", "SPACE", "FN", "MENU",
            "LEFT", "DOWN", "RIGHT"
        ]
        
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.tranfromation_matrix_cam_to_world_wrt_cam = None
        
        self.input = None
        self.model = YOLO('yolov8x.pt')
        
    def setup_robot(self):
        """
        Initialize robot model and visualization
        """
        # Get the path to the URDF file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        urdf_path = os.path.join(script_dir, "../urdf/new_Arm_Urdf.urdf")
        
        # Build the robot model
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        
        # Load visualization models
        self.visual_model = pin.buildGeomFromUrdf(self.model, urdf_path, pin.GeometryType.VISUAL)
        self.collision_model = pin.buildGeomFromUrdf(self.model, urdf_path, pin.GeometryType.COLLISION)
        
        # Setup visualizer
        self.viz = MeshcatVisualizer(self.model, self.collision_model, self.visual_model)
        self.viz.initViewer(loadModel=True)
        
        # Get end effector frame
        self.end_effector_frame = self.model.getFrameId("Link_6")  # Adjust frame name if needed
        
        # Get joint limits
        self.q_min = self.model.lowerPositionLimit
        self.q_max = self.model.upperPositionLimit
        
    def setup_ui(self):
        """
        Setup the user interface
        """
        # Create main layout
        layout = QVBoxLayout()
        
        # Add calibration button
        self.calibrate_btn = QPushButton('Calibrate Keyboard')
        self.calibrate_btn.clicked.connect(self.start_calibration)
        layout.addWidget(self.calibrate_btn)
        
        # Add start typing button
        self.start_btn = QPushButton('Start Typing')
        self.start_btn.clicked.connect(self.start_typing)
        layout.addWidget(self.start_btn)
        
        # Add home position button
        self.home_btn = QPushButton('Go Home')
        self.home_btn.clicked.connect(self.move_to_home)
        layout.addWidget(self.home_btn)
        
        # Set the layout
        self.setLayout(layout)
        self.setWindowTitle('Keyboard Typer IK Controller')
        self.resize(300, 200)
        
    def setup_control(self):
        """
        Initialize control parameters and timers
        """
        # Control parameters
        self.velocity_scale = 0.1  # Velocity scaling factor
        self.dt = 0.05            # Time step for integration
        self.damping = 1e-6       # Regularization factor
        
        # Joint velocity limits
        self.theta_dot_max = 1.0 * np.ones(self.model.nv)
        self.theta_dot_min = -1.0 * np.ones(self.model.nv)
        
        # Initialize joint configuration to neutral position
        self.q = pin.neutral(self.model)
        self.viz.display(self.q)
        
        # Setup control timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.control_loop)
        self.timer.start(int(self.dt * 1000))
        
        self.velocity_scale = 0.1
        self.dt = 0.05
        self.damping = 1e-6
        
    def publish_joint_angles(self, joint_angles):
        """
        Publish joint angles to ROS
        """
        trajectory_msg = JointTrajectory()
        trajectory_msg.joint_names = ['Joint_1', 'Joint_2', 'Joint_3',
                                    'Joint_4', 'Joint_5', 'Joint_6']
        
        point = JointTrajectoryPoint()
        point.positions = joint_angles
        point.time_from_start = rospy.Duration(0.1)
        
        trajectory_msg.points = [point]
        self.pub.publish(trajectory_msg)
            

    def detection(self,image_path, keyboard_dims=(30, 10),padding=50):  # dimensions in cm
        """
        Detect and correct keyboard orientation in an image
        Args:
            self: Object of the class
            image_path: Path to input image or frame
            keyboard_dims: Real-world keyboard dimensions (width, height) in cm
        Returns:
            corners: List of 4 corner points in original image coordinates
            rotated_bbox: Rotated bounding box coordinates
            angle: Detected rotation angle
        """
        # Read image
        img = cv2.imread(image_path)
        # original = img.copy()
        
        # First YOLO detection
        results = self.model(img)
        
        # Get keyboard detection (assuming class index for keyboard is known)
        keyboard_detections = [box for box in results[0].boxes if box.cls == 66]  # 66 is keyboard class in COCO
        
        if not keyboard_detections:
            return None, None, None
        
        # Get the first keyboard detection
        bbox = keyboard_detections[0].xyxy[0].cpu().numpy()  # Convert to numpy array
        x1, y1, x2, y2 = map(int, bbox)
        
        # Get center of bounding box
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        
        # Crop region around keyboard with padding
        crop = img[max(0, y1-padding):min(img.shape[0], y2+padding), 
                max(0, x1-padding):min(img.shape[1], x2+padding)]
        
        # Preprocess for line detection
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Detect lines using Hough Transform
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        if lines is None:
            return None, None, None
        
        # Find dominant angles
        angles = []
        for rho, theta in lines[:, 0]:
            angle = theta * 180 / np.pi
            # Convert to 0-90 degree range
            if angle > 90:
                angle = 180 - angle
            angles.append(angle)
        
        # Find most common angle using histogram
        hist, bins = np.histogram(angles, bins=90)
        dominant_angle = bins[np.argmax(hist)]
        
        # Determine if keyboard is horizontal or vertical based on aspect ratio
        width = x2 - x1
        height = y2 - y1
        aspect_ratio = width / height
        expected_ratio = keyboard_dims[0] / keyboard_dims[1]
        
        # Adjust angle if needed
        if (aspect_ratio > 1) != (expected_ratio > 1):
            dominant_angle += 90
            
        # Rotate image
        rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), dominant_angle, 1.0)
        rotated = cv2.warpAffine(img, rotation_matrix, (img.shape[1], img.shape[0]))
        
        # Second YOLO detection on rotated image
        rotated_results = self.model(rotated)
        rotated_detections = [box for box in rotated_results[0].boxes if box.cls == 66]
        
        if not rotated_detections:
            return None, None, None
            
        # Get refined bounding box
        refined_bbox = rotated_detections[0].xyxy[0].cpu().numpy()
        rx1, ry1, rx2, ry2 = map(int, refined_bbox)
        
        # Get corners of rotated bbox
        corners = np.array([
            [rx1, ry1],
            [rx2, ry1],
            [rx2, ry2],
            [rx1, ry2]
        ], dtype=np.float32)
        
        # Rotate corners back to original orientation
        inv_rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), -dominant_angle, 1.0)
        ones = np.ones(shape=(len(corners), 1))
        corners_homogeneous = np.hstack([corners, ones])
        original_corners = inv_rotation_matrix.dot(corners_homogeneous.T).T
        
        return original_corners, corners, dominant_angle
    
    # TODO : This is not integrated in self.detection() function
    def image_preprocess(self, image_path):
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Resize the image
        resized = cv2.resize(image, None, fx=self.resize_factor, fy=self.resize_factor, interpolation=cv2.INTER_CUBIC)

        # Convert to grayscale
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to estimate the background
        blur = cv2.GaussianBlur(gray, (self.blur_kernel_size, self.blur_kernel_size), 0)

        # Subtract the background
        background_subtracted = cv2.subtract(gray, blur)

        # Normalize the image
        normalized = cv2.normalize(background_subtracted, None, 0, 255, cv2.NORM_MINMAX)

        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=self.clahe_clip, tileGridSize=self.clahe_grid)
        clahe_result = clahe.apply(normalized)

        # Estimate the illumination
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.illumination_kernel, self.illumination_kernel))
        illumination = cv2.morphologyEx(clahe_result, cv2.MORPH_CLOSE, kernel)

        # Correct the illumination
        illumination_corrected = cv2.divide(clahe_result, illumination, scale=255)

        # Binarize the image
        _, binarized = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return {
            "resized": resized,
            "gray": gray,
            "background_subtracted": background_subtracted,
            "normalized": normalized,
            "clahe_result": clahe_result,
            "illumination_corrected": illumination_corrected,
            "binarized": binarized
        }
    
    # TODO : Logic error in the function
    def estimate(self, corner_points, keyboard_dimensions, depth_image):
        """
        Estimates the keyboard's depth, overlays points, and calculates transformations.
        
        :param corner_points: (4x2 array) The detected corner points of the keyboard in the image.
        :param keyboard_dimensions: (width, height) Dimensions of the keyboard in real-world units (e.g., cm).
        :param depth_image: The depth image captured from the depth camera.
        
        :return: A dictionary containing:
                 - 2D points in the image.
                 - 3D world positions of the keyboard corners.
                 - Transformation matrix (camera to world frame).
        """
        # Ensure the corner points are a NumPy array
        corner_points = np.array(corner_points, dtype=np.float32)

        # Get depth values at the corner points
        depths = [depth_image[int(pt[1]), int(pt[0])] for pt in corner_points]

        # Calculate the average depth of the keyboard
        avg_depth = np.mean(depths)

        # Define the real-world coordinates of the keyboard's corners (relative to top-left)
        width, height = keyboard_dimensions
        object_points = np.array([
            [0, 0, 0],  # Top-left
            [width, 0, 0],  # Top-right
            [width, height, 0],  # Bottom-right
            [0, height, 0]  # Bottom-left
        ], dtype=np.float32)

        # Solve for the pose (camera-to-world transformation)
        success, rvec, tvec = cv2.solvePnP(object_points, corner_points, self.camera_matrix, self.dist_coeffs)
        if not success:
            raise ValueError("Unable to solve for pose. Check inputs or calibration data.")

        # Convert rotation vector to matrix
        rotation_matrix, _ = cv2.Rodrigues(rvec)

        # Construct the transformation matrix
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = rotation_matrix
        transformation_matrix[:3, 3] = tvec.ravel()

        # Overlay pre-stored points (relative to top-left corner)
        overlay_points = np.array([
            [10, 10, 0],  # Example point (10 cm right, 10 cm down)
            [20, 10, 0],  # Another example
        ], dtype=np.float32)
        overlay_image_points, _ = cv2.projectPoints(overlay_points, rvec, tvec, self.camera_matrix, self.dist_coeffs)

        # Convert projected points to 2D
        overlay_image_points = overlay_image_points.squeeze(axis=1)
        
        #TODO : Returning bullshit
        return {
            "2D_points": corner_points,  # Original 2D points in the image
            "3D_positions": object_points + np.array([0, 0, avg_depth]),  # Adjusted for average depth
            "camera_to_world_transform": transformation_matrix,
            "overlay_2D": overlay_image_points
        }
    
    # TODO : This function is also bullshit
    def scanning(self, image_path):
        
        corners, _, _ = self.detection(image_path, keyboard_dims=(30, 10), padding=50)
        
        pos2d,pos3d,tf_cw = self.estimate(corners, (30, 10), image_path)
        
        self.key_positions = {str(i): pos3d[i] for i in range(4)}
        self.tranfromation_matrix_cam_to_world_wrt_cam = tf_cw
        
        
    def string_to_keyboard_clicks(input_string):
        keyboard_clicks = []
        caps_active = False  # Track CAPS state

        for char in input_string:
            if char.isupper() and not caps_active:
                # Activate CAPS if the character is uppercase and CAPS is not active
                keyboard_clicks.append("CAPSLOCK")
                caps_active = True
            elif char.islower() and caps_active:
                # Deactivate CAPS if the character is lowercase and CAPS is active
                keyboard_clicks.append("CAPSLOCK")
                caps_active = False
            
            if char.isalnum() or char in {'-', '_'}:  # Letters, numbers, and some symbols
                keyboard_clicks.append(char.upper() if not caps_active else char)
            elif char.isspace():
                keyboard_clicks.append("SPACE")
            else:
                # Add any non-alphanumeric, non-space character as is
                keyboard_clicks.append(char)
        
        # End with ENTER
        keyboard_clicks.append("ENTER")
        
        return keyboard_clicks
    
    def move_to_key(self, target_position: np.ndarray):
        if self.check_workspace_limits(target_position):
            self.set_target(target_position)
            
            # Wait for movement to complete
            while hasattr(self, 'current_target'):
                rospy.sleep(0.01)
        else:
            print("Target position out of workspace bounds")

    def move_to_home(self):
        """
        Move the robot to its home position
        """
        if self.home_position is None:
            # Define default home position if none is set
            self.home_position = np.array([0.2, 0, 0.3])  # Adjust these coordinates as needed
        
        self.move_to_key(self.home_position)
        
    def adjustment(self):
        """
        Fine-tune position before pressing key using visual feedback
        """
        # Get current end effector position
        pin.forwardKinematics(self.model, self.data, self.q)
        pin.updateFramePlacements(self.model, self.data)
        current_position = self.data.oMf[self.end_effector_frame].translation
        
        # TODO: Implement vision-based position correction
        # This could use camera feedback to ensure precise positioning
        pass

    def confirmation(self):
        """
        Confirm successful key press through feedback
        """
        # TODO: Implement key press confirmation
        # This could use force feedback, visual confirmation, or sound detection
        pass

    def control_loop(self):
        """
        Main control loop for robot movement
        Checks key positions and executes movements based on current state
        """
        # Skip if no key positions are defined
        if not self.key_positions:
            return
            
        try:
            # Get current robot state
            pin.forwardKinematics(self.model, self.data, self.q)
            pin.updateFramePlacements(self.model, self.data)
            current_position = self.data.oMf[self.end_effector_frame].translation
            
            # Safety check - ensure we're not too close to workspace limits
            if not self.check_workspace_limits(current_position):
                print("Warning: Approaching workspace limits")
                self.emergency_stop()
                return
                
            # Check if we're currently executing a movement
            if hasattr(self, 'current_target'):
                # Get distance to target
                error = self.current_target - current_position
                
                if np.linalg.norm(error) < 1e-3:  # Within tolerance
                    # Movement complete
                    delattr(self, 'current_target')
                else:
                    # Continue movement to target
                    # Compute Jacobian
                    J = pin.computeFrameJacobian(self.model, self.data, self.q, 
                                            self.end_effector_frame,
                                            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3, :]
                    
                    # QP Setup for smooth motion
                    H = J.T @ J + self.damping * np.eye(self.model.nv)
                    g = -J.T @ error
                    
                    # Joint limits constraints
                    q_upper_violation = (self.q_max - self.q) / self.dt
                    q_lower_violation = (self.q_min - self.q) / self.dt
                    
                    C = np.vstack([np.eye(self.model.nv), -np.eye(self.model.nv),
                                np.eye(self.model.nv), -np.eye(self.model.nv)])
                    b = np.hstack([self.theta_dot_min, -self.theta_dot_max,
                                q_lower_violation, -q_upper_violation])
                    
                    # Solve QP for joint velocities
                    try:
                        dq = solve_qp(H, g, C.T, b)[0]
                        
                        # Scale velocities if needed
                        vel_scale = min(1.0, self.velocity_scale / np.max(np.abs(dq)))
                        dq *= vel_scale
                        
                        # Update configuration
                        self.q = pin.integrate(self.model, self.q, dq * self.dt)
                        
                        # Update visualization and send to robot
                        self.viz.display(self.q)
                        self.publish_joint_angles(self.q)
                        
                    except Exception as e:
                        print(f"QP solver error: {e}")
                        self.emergency_stop()
                        
        except Exception as e:
            print(f"Control loop error: {e}")
            self.emergency_stop()

    # Add this helper method to set movement targets
    def set_target(self, target_position):
        """
        Set a new target position for the control loop
        """
        if self.check_workspace_limits(target_position):
            self.current_target = target_position
        else:
            print("Target position outside workspace limits")
            
    def load_key_positions(self):
        """
        Load saved key positions from file
        """
        try:
            if os.path.exists(self.positions_file):
                self.key_positions = np.load(self.positions_file, allow_pickle=True).item()
                print("Key positions loaded successfully")
                return True
            return False
        except Exception as e:
            print(f"Error loading key positions: {e}")
            return False

    def save_key_positions(self):
        """
        Save key positions to file
        """
        try:
            np.save(self.positions_file, self.key_positions)
            print("Key positions saved successfully")
            return True
        except Exception as e:
            print(f"Error saving key positions: {e}")
            return False
    
    def emergency_stop(self):
        """
        Emergency stop function
        """
        self.timer.stop()
        # Stop all movement
        self.publish_joint_angles(self.q)
        print("Emergency stop activated!")
        
    def check_workspace_limits(self, position):
        """
        Check if target position is within robot workspace
        """
        # Define workspace limits
        workspace_limits = {
            'x': (-0.5, 0.5),  # meters
            'y': (-0.5, 0.5),
            'z': (0.0, 0.6)
        }
        
        return (workspace_limits['x'][0] <= position[0] <= workspace_limits['x'][1] and
                workspace_limits['y'][0] <= position[1] <= workspace_limits['y'][1] and
                workspace_limits['z'][0] <= position[2] <= workspace_limits['z'][1])

            
        
    
        
    