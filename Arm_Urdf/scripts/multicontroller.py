#!/usr/bin/env python3

import pinocchio as pin
import numpy as np
import rospy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QComboBox, QLabel
from PyQt5.QtCore import Qt, QTimer
from quadprog import solve_qp
import os
from pinocchio.visualize import MeshcatVisualizer
from enum import Enum, auto
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import logging
import threading
from queue import Queue
import time

@dataclass
class SharedResources:
    """Shared resources between controllers"""
    model: pin.Model
    data: pin.Data
    viz: Optional[MeshcatVisualizer]
    urdf_path: str
    end_effector_frame: str
    control_frequency: float = 200.0

class BaseController:
    """Base class for all controllers"""
    
    def __init__(self, shared_resources: SharedResources):
        self.shared_resources = shared_resources
        self.model = shared_resources.model
        self.data = shared_resources.data
        
        # Initialize state variables
        self.q = pin.neutral(self.model)
        self.dq = np.zeros(self.model.nv)
        self.ddq = np.zeros(self.model.nv)
        
        self.active = False
        self.command_queue = Queue()
        
    def activate(self):
        """Activate this controller"""
        self.active = True
        
    def deactivate(self):
        """Deactivate this controller"""
        self.active = False
        self.dq = np.zeros(self.model.nv)
        self.ddq = np.zeros(self.model.nv)
        
    def update(self):
        """Update controller state"""
        if not self.active:
            return
            
        # Process commands from queue
        while not self.command_queue.empty():
            cmd = self.command_queue.get()
            self.process_command(cmd)
            
        # Compute and apply control
        control = self.compute_control()
        self.apply_control(control)
        
    def process_command(self, cmd: Dict):
        """Process a command from the queue"""
        pass
        
    def compute_control(self) -> np.ndarray:
        """Compute control output"""
        raise NotImplementedError
        
    def apply_control(self, control: np.ndarray):
        """Apply computed control"""
        dt = 1.0 / self.shared_resources.control_frequency
        self.dq = control
        self.q = pin.integrate(self.model, self.q, self.dq * dt)
        
    def get_current_configuration(self) -> np.ndarray:
        """Get current joint configuration"""
        return self.q
        
    def stop(self):
        """Stop the controller"""
        self.deactivate()

class VelocityController(BaseController):
    """Velocity-based controller with keyboard control"""
    
    def __init__(self, shared_resources: SharedResources):
        super().__init__(shared_resources)
        self.target_velocity = None
        self.target_position = None
        self.velocity_scale = 0.1
        self.pressed_keys = set()
        
        # Initialize ROS publisher
        self.trajectory_pub = rospy.Publisher('/joint_trajectory', JointTrajectory, queue_size=10)
        
        # Keyboard mapping for Cartesian velocities
        self.key_twist_mapping = {
            Qt.Key_W: np.array([self.velocity_scale, 0, 0, 0, 0, 0]),  # Forward
            Qt.Key_S: np.array([-self.velocity_scale, 0, 0, 0, 0, 0]), # Backward
            Qt.Key_A: np.array([0, self.velocity_scale, 0, 0, 0, 0]),  # Left
            Qt.Key_D: np.array([0, -self.velocity_scale, 0, 0, 0, 0]), # Right
            Qt.Key_Q: np.array([0, 0, self.velocity_scale, 0, 0, 0]),  # Up
            Qt.Key_E: np.array([0, 0, -self.velocity_scale, 0, 0, 0]), # Down
            Qt.Key_J: np.array([0, 0, 0, -self.velocity_scale, 0, 0]), # Rotate X left
            Qt.Key_L: np.array([0, 0, 0, self.velocity_scale, 0, 0]),  # Rotate X right
            Qt.Key_I: np.array([0, 0, 0, 0, self.velocity_scale, 0]),  # Rotate Y down
            Qt.Key_K: np.array([0, 0, 0, 0, -self.velocity_scale, 0]), # Rotate Y up
            Qt.Key_U: np.array([0, 0, 0, 0, 0, self.velocity_scale]),  # Yaw left
            Qt.Key_O: np.array([0, 0, 0, 0, 0, -self.velocity_scale])  # Yaw right
        }
        
    def handle_keyboard_input(self, pressed_keys: set):
        """Handle keyboard input and compute Cartesian velocity"""
        self.pressed_keys = pressed_keys
        
        # Combine all active key velocities
        total_velocity = np.zeros(6)
        for key in self.pressed_keys:
            if key in self.key_twist_mapping:
                total_velocity += self.key_twist_mapping[key]
                
        return total_velocity
        
    def compute_control(self) -> np.ndarray:
        """Compute control based on keyboard input"""
        # Get Cartesian velocity from keyboard
        cartesian_velocity = self.handle_keyboard_input(self.pressed_keys)
        
        # If no keys are pressed, return zero velocity
        if np.all(cartesian_velocity == 0):
            return np.zeros(self.model.nv)
            
        # Update forward kinematics
        pin.forwardKinematics(self.model, self.data, self.q)
        pin.updateFramePlacements(self.model, self.data)
        
        # Compute Jacobian
        J = pin.computeFrameJacobian(
            self.model, self.data, self.q,
            self.shared_resources.end_effector_frame,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        
        # Compute joint velocities using pseudoinverse
        dq = np.linalg.pinv(J) @ cartesian_velocity
        
        # Publish joint angles to ROS
        self.publish_joint_angles(self.q)
        
        return dq
        
    def publish_joint_angles(self, joint_angles):
        """Publish joint angles to ROS"""
        trajectory_msg = JointTrajectory()
        trajectory_msg.joint_names = ['Joint_1', 'Joint_2', 'Joint_3',
                                    'Joint_4', 'Joint_5', 'Joint_6']
        
        point = JointTrajectoryPoint()
        point.positions = joint_angles.tolist()
        point.time_from_start = rospy.Duration(0.1)
        
        trajectory_msg.points = [point]
        self.trajectory_pub.publish(trajectory_msg)

class PositionController(BaseController):
    """Position-based controller"""
    
    def __init__(self, shared_resources: SharedResources):
        super().__init__(shared_resources)
        self.target_position = None
        self.kp = 100.0  # Position gain
        self.kd = 20.0   # Velocity gain
        
    def set_target(self, position: np.ndarray):
        self.target_position = position
        
    def compute_control(self) -> np.ndarray:
        if self.target_position is None:
            return np.zeros(self.model.nv)
            
        error = self.target_position - self.q
        velocity_error = -self.dq
        return self.kp * error + self.kd * velocity_error

class ImpedanceController(BaseController):
    """Impedance-based controller"""
    
    def __init__(self, shared_resources: SharedResources):
        super().__init__(shared_resources)
        self.target_position = None
        self.stiffness = np.eye(6) * 1000
        self.damping = np.eye(6) * 50
        
    def set_target(self, position: np.ndarray):
        self.target_position = position
        
    def compute_control(self) -> np.ndarray:
        if self.target_position is None:
            return np.zeros(self.model.nv)
            
        pin.forwardKinematics(self.model, self.data, self.q)
        current_position = self.data.oMf[self.shared_resources.end_effector_frame].translation
        
        position_error = self.target_position - current_position
        J = pin.computeFrameJacobian(
            self.model, self.data, self.q,
            self.shared_resources.end_effector_frame,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        
        f_imp = self.stiffness @ position_error
        return J.T @ f_imp

class MasterController:
    """Master controller managing multiple predefined controllers"""
    
    def __init__(self, urdf_path: str):
        # Initialize ROS node
        rospy.init_node('master_robot_controller', anonymous=True)
        
        # Initialize shared resources
        self.shared_resources = self._initialize_shared_resources(urdf_path)
        
        # Initialize GUI
        self.app = QApplication([])
        self.setup_gui()
        
        # Create controllers dictionary
        self.controllers: Dict[str, BaseController] = {}
        self.active_controller: Optional[str] = None
        
        # Initialize controllers
        self._initialize_controllers()
        
        # Initialize update timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(int(1000/self.shared_resources.control_frequency))
        
        self.pressed_keys = set()
        
        
    def _initialize_shared_resources(self, urdf_path: str) -> SharedResources:
        """Initialize shared resources for all controllers"""
        model = pin.buildModelFromUrdf(urdf_path)
        data = model.createData()
        
        # Initialize visualizer
        try:
            viz = MeshcatVisualizer(model, model.collision_model, model.visual_model)
            viz.initViewer()
            viz.loadViewerModel()
        except Exception as e:
            logging.warning(f"Could not initialize visualizer: {e}")
            viz = None
            
        return SharedResources(
            model=model,
            data=data,
            viz=viz,
            urdf_path=urdf_path,
            end_effector_frame="Link_6"
        )
        
    def _initialize_controllers(self):
        """Initialize predefined controllers"""
        # Create controller instances
        self.controllers = {
            "Velocity Controller": VelocityController(self.shared_resources),
            "Position Controller": PositionController(self.shared_resources),
            "Impedance Controller": ImpedanceController(self.shared_resources)
        }
        
        # Update GUI controller list
        self.controller_combo.addItems(self.controllers.keys())
        
    def setup_gui(self):
        """Setup master GUI with key event handling"""
        self.window = QWidget()
        self.window.setWindowTitle('Master Robot Controller')
        layout = QVBoxLayout()

        # Controller selection
        self.controller_combo = QComboBox()
        self.controller_combo.currentTextChanged.connect(self.switch_active_controller)
        layout.addWidget(QLabel("Active Controller:"))
        layout.addWidget(self.controller_combo)

        # Add keyboard control instructions
        key_instructions = QLabel(
            "Keyboard Controls:\n"
            "W/S: Forward/Backward\n"
            "A/D: Left/Right\n"
            "Q/E: Up/Down\n"
            "J/L: Rotate X\n"
            "I/K: Rotate Y\n"
            "U/O: Rotate Z"
        )
        layout.addWidget(key_instructions)

        self.window.setLayout(layout)
        self.window.setFocusPolicy(Qt.StrongFocus)

        # Set the main window to handle key events
        self.window.keyPressEvent = self.keyPressEvent
        self.window.keyReleaseEvent = self.keyReleaseEvent
        
    def switch_active_controller(self, controller_name: str):
        """Switch the active controller"""
        if controller_name in self.controllers:
            if self.active_controller:
                self.controllers[self.active_controller].deactivate()
            self.active_controller = controller_name
            self.controllers[controller_name].activate()
            logging.info(f"Switched to controller: {controller_name}")
            
    def keyPressEvent(self, event):
        """Handle key press events"""
        self.pressed_keys.add(event.key())
        if self.active_controller and isinstance(self.controllers[self.active_controller], VelocityController):
            self.controllers[self.active_controller].pressed_keys = self.pressed_keys
            
    def keyReleaseEvent(self, event):
        """Handle key release events"""
        self.pressed_keys.discard(event.key())
        if self.active_controller and isinstance(self.controllers[self.active_controller], VelocityController):
            self.controllers[self.active_controller].pressed_keys = self.pressed_keys
            
    def update(self):
        """Update active controller and visualization"""
        if self.active_controller and self.active_controller in self.controllers:
            # Update controller
            self.controllers[self.active_controller].update()
            
            # Update visualization
            if self.shared_resources.viz is not None:
                current_q = self.controllers[self.active_controller].get_current_configuration()
                self.shared_resources.viz.display(current_q)
    
    def set_target(self, position: np.ndarray):
        """Set target for current active controller"""
        if self.active_controller:
            self.controllers[self.active_controller].set_target(position)
            
    def start(self):
        """Start the master controller"""
        self.window.show()
        return self.app.exec_()
        
    def stop(self):
        """Stop all controllers and cleanup"""
        for controller in self.controllers.values():
            controller.stop()
        self.app.quit()

def main():
    """Main function"""
    # Get URDF path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    urdf_path = os.path.join(script_dir, "../urdf/Arm_Urdf.urdf")
    
    # Create master controller
    master = MasterController(urdf_path)
    
    # Example of setting a target
    # target = np.array([0.5, 0.0, 0.5])  # Example target position
    # master.set_target(target)
    
    # Start the application
    return master.start()

if __name__ == "__main__":
    main()