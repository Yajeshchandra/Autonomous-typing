# #!/usr/bin/env python3

# import pinocchio as pin
# import numpy as np
# from sklearn.linear_model import LinearRegression
# import rospy
# from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
# from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout
# from PyQt5.QtCore import Qt, QTimer
# from quadprog import solve_qp
# import os
# from pinocchio.visualize import MeshcatVisualizer
# from typing import Dict, List, Tuple
# from pynput.keyboard import Key, KeyCode, Listener

# class KeyboardTyperWithIK(QWidget):
#     def __init__(self):
#         super().__init__()
#         self.setup_robot()
#         self.setup_ui()
#         self.setup_control()
        
#         # Keyboard typer specific attributes
#         self.key_positions: Dict[str, np.ndarray] = {}
#         self.home_position: np.ndarray = None
#         self.hover_distance = 0.01  # 1cm hover distance
#         self.positions_file = "keyboard_positions.npy"
#         self.recording_mode = False
#         self.current_key = None
        
#         # Plane fitting attributes
#         self.plane_normal = None
#         self.plane_point = None
#         self.hover_plane_normal = None
#         self.hover_plane_point = None
        
#         # Control parameters
#         self.velocity_scale = 0.1
#         self.dt = 0.05
#         self.damping = 1e-6
        
#         # Initialize keyboard listener for recording mode
#         self.keyboard_listener = None

#     def setup_robot(self):
#         """Initialize robot model and visualization"""
#         urdf_path = "/home/ub20/rsws/src/iiwa_description/urdf/iiwa7.urdf"
#         self.model = pin.buildModelFromUrdf(urdf_path)
#         self.data = self.model.createData()
        
#         # Visualization setup
#         visual_model = pin.buildGeomFromUrdf(self.model, urdf_path, pin.GeometryType.VISUAL)
#         collision_model = pin.buildGeomFromUrdf(self.model, urdf_path, pin.GeometryType.COLLISION)
#         self.viz = MeshcatVisualizer(self.model, collision_model, visual_model)
#         self.viz.initViewer(loadModel=True)
        
#         # End-effector frame
#         self.end_effector_frame = self.model.getFrameId("iiwa_link_7")
        
#         # Initialize joint configuration
#         self.q = pin.neutral(self.model)
#         self.viz.display(self.q)
        
#         # Joint limits
#         self.q_min = self.model.lowerPositionLimit
#         self.q_max = self.model.upperPositionLimit
        
#         # ROS setup
#         self.pub = rospy.Publisher('/body_controller/command', JointTrajectory, queue_size=10)

#     def setup_ui(self):
#         """Setup the user interface"""
#         layout = QVBoxLayout()
        
#         # Add buttons for different modes
#         self.scan_button = QPushButton('Start Keyboard Scanning')
#         self.scan_button.clicked.connect(self.toggle_scanning_mode)
#         layout.addWidget(self.scan_button)
        
#         self.type_button = QPushButton('Type Text')
#         self.type_button.clicked.connect(self.start_typing)
#         layout.addWidget(self.type_button)
        
#         self.setLayout(layout)

#     def setup_control(self):
#         """Setup control loop timer"""
#         self.pressed_keys = set()
#         self.timer = QTimer()
#         self.timer.timeout.connect(self.control_loop)
#         self.timer.start(int(self.dt * 1000))

#     def toggle_scanning_mode(self):
#         """Toggle keyboard scanning mode with plane fitting"""
#         self.recording_mode = not self.recording_mode
#         if self.recording_mode:
#             self.scan_button.setText('Stop Scanning')
#             self.start_keyboard_listener()
#             self.home_position = self.q.copy()
#             print("Recording mode started. Home position saved.")
#         else:
#             self.scan_button.setText('Start Keyboard Scanning')
#             self.stop_keyboard_listener()
#             self.save_positions()
#             self.visualize_planes()
#             print("Recording mode stopped. Positions saved and planes fitted.")

#     def start_keyboard_listener(self):
#         """Start the keyboard listener for recording mode"""
#         self.keyboard_listener = Listener(on_press=self.on_record_key)
#         self.keyboard_listener.start()

#     def stop_keyboard_listener(self):
#         """Stop the keyboard listener"""
#         if self.keyboard_listener:
#             self.keyboard_listener.stop()

#     def on_record_key(self, key):
#         """Handle key press during recording mode"""
#         if not self.recording_mode:
#             return

#         if key == Key.esc:
#             self.toggle_scanning_mode()
#             return

#         try:
#             # Convert key to string representation
#             if hasattr(key, 'char'):
#                 key_str = key.char.upper()
#             elif key == Key.space:
#                 key_str = 'SPACE'
#             elif key == Key.caps_lock:
#                 key_str = 'CAPSLOCK'
#             else:
#                 return

#             # Save current position for this key
#             self.key_positions[key_str] = self.get_end_effector_position()
#             print(f"Position recorded for key: {key_str}")
            
#         except Exception as e:
#             print(f"Error recording key: {e}")

#     def get_end_effector_position(self) -> np.ndarray:
#         """Get current end effector position"""
#         pin.forwardKinematics(self.model, self.data, self.q)
#         pin.updateFramePlacements(self.model, self.data)
#         return self.data.oMf[self.end_effector_frame].translation.copy()

#     def fit_keyboard_plane(self):
#         """Fit a plane to the recorded key positions using linear regression"""
#         if len(self.key_positions) < 3:
#             print("Need at least 3 points to fit a plane")
#             return False

#         # Convert positions to numpy array
#         points = np.array([pos for pos in self.key_positions.values()])
        
#         # Use XY coordinates to predict Z coordinates
#         X = points[:, :2]  # XY coordinates
#         y = points[:, 2]   # Z coordinates
        
#         # Fit plane using linear regression
#         reg = LinearRegression()
#         reg.fit(X, y)
        
#         # Calculate plane normal and point
#         self.plane_normal = np.array([reg.coef_[0], reg.coef_[1], -1])
#         self.plane_normal = self.plane_normal / np.linalg.norm(self.plane_normal)
        
#         # Use mean point as a point on the plane
#         self.plane_point = np.mean(points, axis=0)
        
#         # Calculate hover plane (parallel plane above keyboard)
#         self.hover_plane_normal = self.plane_normal
#         self.hover_plane_point = self.plane_point + self.hover_distance * self.plane_normal
        
#         print("Keyboard plane fitted successfully")
#         print(f"Plane normal: {self.plane_normal}")
#         print(f"Plane point: {self.plane_point}")
#         return True

#     def save_positions(self):
#         """Save recorded positions and plane parameters to file"""
#         # First fit the plane
#         self.fit_keyboard_plane()
        
#         np.save(self.positions_file, {
#             'key_positions': self.key_positions,
#             'home_position': self.home_position,
#             'plane_normal': self.plane_normal,
#             'plane_point': self.plane_point,
#             'hover_plane_normal': self.hover_plane_normal,
#             'hover_plane_point': self.hover_plane_point
#         })

#     def load_positions(self) -> bool:
#         """Load stored keyboard positions and plane parameters"""
#         try:
#             data = np.load(self.positions_file, allow_pickle=True).item()
#             self.key_positions = data['key_positions']
#             self.home_position = data['home_position']
#             self.plane_normal = data['plane_normal']
#             self.plane_point = data['plane_point']
#             self.hover_plane_normal = data['hover_plane_normal']
#             self.hover_plane_point = data['hover_plane_point']
#             return True
#         except Exception as e:
#             print(f"Error loading positions: {e}")
#             return False

#     def project_point_to_hover_plane(self, point: np.ndarray) -> np.ndarray:
#         """Project a point onto the hover plane"""
#         if self.hover_plane_normal is None or self.hover_plane_point is None:
#             # Fallback to simple vertical offset if plane isn't fitted
#             return point + np.array([0, 0, self.hover_distance])
            
#         # Vector from hover plane point to the target point
#         v = point - self.hover_plane_point
        
#         # Project v onto the plane normal to get the distance to the plane
#         dist = np.dot(v, self.hover_plane_normal)
        
#         # Project the point onto the hover plane
#         projected_point = point - dist * self.hover_plane_normal
        
#         return projected_point

#     def get_hover_position(self, key_position: np.ndarray) -> np.ndarray:
#         """Calculate hover position using fitted plane"""
#         if self.hover_plane_normal is None:
#             # Fallback to simple vertical offset if plane isn't fitted
#             return key_position + np.array([0, 0, self.hover_distance])
            
#         # Project the key position onto the hover plane
#         hover_pos = self.project_point_to_hover_plane(key_position)
        
#         # Add a small offset in the direction of the plane normal for safety
#         safety_margin = 0.005  # 5mm additional safety margin
#         hover_pos += safety_margin * self.hover_plane_normal
        
#         return hover_pos

#     def visualize_planes(self):
#         """Visualize the keyboard and hover planes for debugging"""
#         if self.plane_normal is None:
#             return
            
#         # Create a grid of points around the keyboard area
#         min_pos = np.min([pos for pos in self.key_positions.values()], axis=0)
#         max_pos = np.max([pos for pos in self.key_positions.values()], axis=0)
        
#         x = np.linspace(min_pos[0] - 0.05, max_pos[0] + 0.05, 10)
#         y = np.linspace(min_pos[1] - 0.05, max_pos[1] + 0.05, 10)
#         X, Y = np.meshgrid(x, y)
        
#         # Calculate Z coordinates for both planes
#         keyboard_Z = np.zeros_like(X)
#         hover_Z = np.zeros_like(X)
        
#         for i in range(X.shape[0]):
#             for j in range(X.shape[1]):
#                 point = np.array([X[i,j], Y[i,j], 0])
#                 # Project points onto both planes
#                 keyboard_point = self.project_point_to_hover_plane(point)
#                 hover_point = self.get_hover_position(point)
#                 keyboard_Z[i,j] = keyboard_point[2]
#                 hover_Z[i,j] = hover_point[2]

#         # Visualize using markers or meshes in your viewer
#         self.visualize_plane_mesh(X, Y, keyboard_Z, "keyboard_plane")
#         self.visualize_plane_mesh(X, Y, hover_Z, "hover_plane")

#     def visualize_plane_mesh(self, X, Y, Z, name):
#         """Helper method to visualize a plane mesh"""
#         # Implementation depends on your visualization system
#         # For example, using RViz markers or MeshcatVisualizer
#         pass

#     def start_typing(self):
#         """Start typing sequence"""
#         text = input("Enter text to type: ")
#         self.type_sequence(text)

#     def type_sequence(self, text: str):
#         """Execute typing sequence for given text"""
#         if not self.load_positions():
#             print("No keyboard positions found. Please scan keyboard first.")
#             return

#         keystrokes = self.convert_text_to_keystrokes(text)
#         print(f"Executing keystroke sequence: {keystrokes}")

#         # Move to home position
#         self.move_to_position(self.home_position)

#         for keystroke in keystrokes:
#             if keystroke not in self.key_positions:
#                 print(f"Position for key {keystroke} not found!")
#                 continue

#             key_pos = self.key_positions[keystroke]
#             hover_pos = self.get_hover_position(key_pos)

#             # Execute key press sequence
#             self.move_to_position(hover_pos)
#             rospy.sleep(0.1)  # Short pause at hover position
            
#             self.move_to_position(key_pos)
#             rospy.sleep(0.1)  # Short pause for key press
            
#             self.move_to_position(hover_pos)
#             rospy.sleep(0.1)  # Short pause at hover position

#         # Return to home
#         self.move_to_position(self.home_position)

#     def move_to_position(self, target_position: np.ndarray):
#         """Move to target position using velocity IK"""
#         while True:
#             current_pos = self.get_end_effector_position()
#             error = target_position - current_pos
            
#             if np.linalg.norm(error) < 0.001:  # 1mm threshold
#                 break
                
#             # Compute Jacobian
#             pin.computeJointJacobians(self.model, self.data, self.q)
#             J = pin.computeFrameJacobian(self.model, self.data, self.q, 
#                                        self.end_effector_frame, 
#                                        pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)

#             # Setup QP for IK
#             H = J.T @ J + self.damping * np.eye(self.model.nv)
#             g = -J.T @ error  # Negative because we want to minimize the error

#             # Joint limits constraints
#             q_upper_violation = (self.q_max - self.q) / self.dt
#             q_lower_violation = (self.q_min - self.q) / self.dt
            
#             C = np.vstack([np.eye(self.model.nv), -np.eye(self.model.nv)])
#             b = np.hstack([q_lower_violation, -q_upper_violation])

#             # Solve QP
#             theta_dot = solve_qp(H, g, C.T, b)[0]

#             # Update configuration
#             self.q = pin.integrate(self.model, self.q, theta_dot * self.dt)
#             self.viz.display(self.q)
#             self.publish_joint_angles(self.q)
            
#             rospy.sleep(self.dt)

#     def control_loop(self):
#         """Main control loop for autonomous typing and manual control"""
#         if self.recording_mode:
#             # During recording, visualize current position and plane if fitted
#             if self.plane_normal is not None:
#                 self.visualize_planes()
#             return

#         # Get current end effector pose
#         pin.forwardKinematics(self.model, self.data, self.q)
#         pin.updateFramePlacements(self.model, self.data)
#         current_pos = self.data.oMf[self.end_effector_frame].translation
#         current_rot = self.data.oMf[self.end_effector_frame].rotation

#         # Check if we're currently executing a typing sequence
#         if hasattr(self, 'typing_sequence') and self.typing_sequence:
#             current_target = self.typing_sequence[0]
            
#             # Calculate distance to target
#             distance = np.linalg.norm(current_pos - current_target['position'])
            
#             if distance < 0.001:  # Within 1mm of target
#                 # Handle different phases of key pressing
#                 if current_target['type'] == 'hover':
#                     # At hover position, move to key press
#                     self.typing_sequence.pop(0)  # Remove current target
#                     if self.typing_sequence:  # If there's a next target
#                         print(f"Moving to press key: {current_target['key']}")
                        
#                 elif current_target['type'] == 'press':
#                     # At press position, wait briefly then move back to hover
#                     rospy.sleep(0.1)  # Brief pause for key press
#                     self.typing_sequence.pop(0)  # Remove current target
#                     if self.typing_sequence:
#                         print(f"Completed pressing key: {current_target['key']}")
                        
#                 elif current_target['type'] == 'return_hover':
#                     # Back at hover, ready for next key
#                     self.typing_sequence.pop(0)  # Remove current target
#                     if not self.typing_sequence:  # If sequence complete
#                         print("Typing sequence completed")
#                         self.typing_sequence = None  # Clear sequence
#                         return
                        
#                 return  # Skip velocity calculation this cycle
                
#             # Calculate velocity command using IK
#             error = current_target['position'] - current_pos
            
#             # Compute Jacobian
#             pin.computeJointJacobians(self.model, self.data, self.q)
#             J = pin.computeFrameJacobian(model=self.model,
#                                        data=self.data,
#                                        frame_id=self.end_effector_frame,
#                                        rf_frame=pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            
#             # Setup QP for IK
#             H = J.T @ J + self.damping * np.eye(self.model.nv)
#             g = -J.T @ error  # Negative because we want to minimize the error
            
#             # Joint limits and velocity constraints
#             q_upper_violation = (self.q_max - self.q) / self.dt
#             q_lower_violation = (self.q_min - self.q) / self.dt
            
#             # Velocity limits
#             v_limit = 0.5  # Maximum joint velocity (rad/s)
#             v_constraints = np.array([
#                 [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#                 [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
#                 [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
#                 [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
#                 [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
#                 [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
#             ])
            
#             C = np.vstack([
#                 v_constraints,  # Positive velocity limits
#                 -v_constraints,  # Negative velocity limits
#                 np.eye(self.model.nv),  # Joint position upper limits
#                 -np.eye(self.model.nv)  # Joint position lower limits
#             ])
            
#             b = np.hstack([
#                 v_limit * np.ones(self.model.nv),  # Positive velocity limits
#                 v_limit * np.ones(self.model.nv),  # Negative velocity limits
#                 q_upper_violation,  # Joint position upper limits
#                 -q_lower_violation  # Joint position lower limits
#             ])

#             try:
#                 # Solve QP for joint velocities
#                 theta_dot = solve_qp(H, g, C.T, b)[0]
                
#                 # Scale velocity if needed
#                 velocity_norm = np.linalg.norm(theta_dot)
#                 if velocity_norm > v_limit:
#                     theta_dot *= v_limit / velocity_norm
                
#                 # Update joint configuration
#                 self.q = pin.integrate(self.model, self.q, theta_dot * self.dt)
                
#                 # Update visualization and robot
#                 self.viz.display(self.q)
#                 self.publish_joint_angles(self.q)
                
#             except Exception as e:
#                 print(f"Error solving IK: {e}")
#                 self.typing_sequence = None  # Abort sequence on error
        
#         else:
#             # Manual control mode using keyboard input
#             if self.pressed_keys:
#                 # Calculate desired twist based on pressed keys
#                 twist = self.compute_manual_twist()
#                 if np.linalg.norm(twist) > 0:
#                     # Similar IK solution as above but using twist instead of position error
#                     # Implementation of manual control...
#                     pass

#     def compute_manual_twist(self):
#         """Compute desired twist based on pressed keys"""
#         twist = np.zeros(6)
#         for key in self.pressed_keys:
#             if key in self.key_twist_mapping:
#                 twist += self.key_twist_mapping[key]
#         return twist

#     def start_typing_sequence(self, text: str):
#         """Initialize a typing sequence for autonomous execution"""
#         keystrokes = self.convert_text_to_keystrokes(text)
#         self.typing_sequence = []
        
#         for keystroke in keystrokes:
#             if keystroke not in self.key_positions:
#                 print(f"Position for key {keystroke} not found!")
#                 continue
                
#             key_pos = self.key_positions[keystroke]
#             hover_pos = self.get_hover_position(key_pos)
            
#             # Add hover position
#             self.typing_sequence.append({
#                 'type': 'hover',
#                 'key': keystroke,
#                 'position': hover_pos
#             })
            
#             # Add key press position
#             self.typing_sequence.append({
#                 'type': 'press',
#                 'key': keystroke,
#                 'position': key_pos
#             })
            
#             # Add return to hover
#             self.typing_sequence.append({
#                 'type': 'return_hover',
#                 'key': keystroke,
#                 'position': hover_pos
#             })
        
#         print(f"Initialized typing sequence with {len(self.typing_sequence)} movements")

#     def publish_joint_angles(self, joint_angles):
#         """Publish joint angles to ROS"""
#         trajectory_msg = JointTrajectory()
#         trajectory_msg.joint_names = ['iiwa_joint_1', 'iiwa_joint_2', 'iiwa_joint_3',
#                                     'iiwa_joint_4', 'iiwa_joint_5', 'iiwa_joint_6']
        
#         point = JointTrajectoryPoint()
#         point.positions = joint_angles.tolist()
#         point.time_from_start = rospy.Duration(0.1)
        
#         trajectory_msg.points = [point]
#         self.pub.publish(trajectory_msg)