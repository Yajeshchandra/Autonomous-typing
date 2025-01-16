#!/usr/bin/env python3

import pinocchio as pin
import numpy as np
import rospy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QComboBox, QLabel
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

class ControllerType(Enum):
    """Types of available controllers"""
    VELOCITY = auto()
    POSITION = auto()
    IMPEDANCE = auto()
    HYBRID = auto()

@dataclass
class SharedResources:
    """Shared resources between controllers"""
    model: pin.Model
    data: pin.Data
    viz: Optional[MeshcatVisualizer]
    urdf_path: str
    end_effector_frame: str
    control_frequency: float = 200.0

class MasterController:
    """Master controller managing multiple controller instances"""
    
    def __init__(self, urdf_path: str):
        # Initialize ROS node
        rospy.init_node('master_robot_controller', anonymous=True)
        
        # Initialize shared resources
        self.shared_resources = self._initialize_shared_resources(urdf_path)
        
        # Initialize controllers dictionary
        self.controllers: Dict[str, BaseController] = {}
        self.active_controller: Optional[str] = None
        
        # Initialize GUI
        self.app = QApplication([])
        self.setup_gui()
        
        # Initialize update timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(int(1000/self.shared_resources.control_frequency))
        
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
            end_effector_frame="Link_6"  # Adjust as needed
        )
        
    def setup_gui(self):
        """Setup master GUI"""
        self.window = QWidget()
        self.window.setWindowTitle('Master Robot Controller')
        layout = QVBoxLayout()
        
        # Controller selection
        self.controller_combo = QComboBox()
        self.controller_combo.currentTextChanged.connect(self.switch_active_controller)
        layout.addWidget(QLabel("Active Controller:"))
        layout.addWidget(self.controller_combo)
        
        # Add controller button
        add_controller_combo = QComboBox()
        add_controller_combo.addItems([ct.name for ct in ControllerType])
        layout.addWidget(QLabel("Add Controller:"))
        layout.addWidget(add_controller_combo)
        
        add_button = QPushButton("Add Controller")
        add_button.clicked.connect(
            lambda: self.add_controller(
                add_controller_combo.currentText(),
                f"{add_controller_combo.currentText().lower()}_controller_{len(self.controllers)}"
            )
        )
        layout.addWidget(add_button)
        
        self.window.setLayout(layout)
        
    def add_controller(self, controller_type: str, name: str):
        """Add a new controller instance"""
        if name in self.controllers:
            logging.warning(f"Controller {name} already exists")
            return
            
        controller_class = {
            "VELOCITY": VelocityController,
            "POSITION": PositionController,
            "IMPEDANCE": ImpedanceController,
            "HYBRID": HybridController
        }.get(controller_type)
        
        if controller_class is None:
            logging.error(f"Unknown controller type: {controller_type}")
            return
            
        self.controllers[name] = controller_class(self.shared_resources)
        self.controller_combo.addItem(name)
        logging.info(f"Added {controller_type} controller as {name}")
        
    def switch_active_controller(self, controller_name: str):
        """Switch the active controller"""
        if controller_name in self.controllers:
            if self.active_controller:
                self.controllers[self.active_controller].deactivate()
            self.active_controller = controller_name
            self.controllers[controller_name].activate()
            logging.info(f"Switched to controller: {controller_name}")
            
    def update(self):
        """Update active controller and visualization"""
        if self.active_controller and self.active_controller in self.controllers:
            self.controllers[self.active_controller].update()
            
            # Update visualization
            if self.shared_resources.viz is not None:
                current_q = self.controllers[self.active_controller].get_current_configuration()
                self.shared_resources.viz.display(current_q)
                
    def start(self):
        """Start the master controller"""
        self.window.show()
        return self.app.exec_()
        
    def stop(self):
        """Stop all controllers and cleanup"""
        for controller in self.controllers.values():
            controller.stop()
        self.app.quit()

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
        pass  # Implement in derived classes
        
    def compute_control(self) -> np.ndarray:
        """Compute control output"""
        pass  # Implement in derived classes
        
    def apply_control(self, control: np.ndarray):
        """Apply computed control"""
        pass  # Implement in derived classes
        
    def get_current_configuration(self) -> np.ndarray:
        """Get current joint configuration"""
        return self.q
        
    def stop(self):
        """Stop the controller"""
        self.deactivate()

class VelocityController(BaseController):
    """Velocity-based controller"""
    
    def __init__(self, shared_resources: SharedResources):
        super().__init__(shared_resources)
        self.target_velocity = None
        
    def compute_control(self) -> np.ndarray:
        if self.target_velocity is None:
            return np.zeros(self.model.nv)
            
        # Implement velocity control logic here
        return self.target_velocity

class PositionController(BaseController):
    """Position-based controller"""
    
    def __init__(self, shared_resources: SharedResources):
        super().__init__(shared_resources)
        self.target_position = None
        self.kp = 100.0  # Position gain
        self.kd = 20.0   # Velocity gain
        
    def compute_control(self) -> np.ndarray:
        if self.target_position is None:
            return np.zeros(self.model.nv)
            
        # Implement position control logic here
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
        
    def compute_control(self) -> np.ndarray:
        if self.target_position is None:
            return np.zeros(self.model.nv)
            
        # Implement impedance control logic here
        # This is a simplified version
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

class HybridController(BaseController):
    """Hybrid position/force controller"""
    
    def __init__(self, shared_resources: SharedResources):
        super().__init__(shared_resources)
        # Implement hybrid control specific initialization

def main():
    """Main function"""
    # Get URDF path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    urdf_path = os.path.join(script_dir, "../urdf/new_Arm_Urdf.urdf")
    
    # Create and start master controller
    master = MasterController(urdf_path)
    
    # Add some initial controllers
    master.add_controller("VELOCITY", "velocity_controller_0")
    master.add_controller("POSITION", "position_controller_0")
    
    # Start the application
    return master.start()

if __name__ == "__main__":
    main()