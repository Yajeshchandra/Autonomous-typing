#!/usr/bin/env python3

import pinocchio as pin
import numpy as np
from abc import ABC, abstractmethod
import rospy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Any
import logging
from scipy.spatial.transform import Rotation
import threading
from queue import Queue
import time

class ControllerState(Enum):
    """Enum for controller states"""
    INACTIVE = auto()
    INITIALIZING = auto()
    RUNNING = auto()
    PAUSED = auto()
    ERROR = auto()
    EMERGENCY_STOP = auto()
    HOMING = auto()
    CALIBRATING = auto()

@dataclass
class ControllerConfig:
    """Enhanced configuration for controller parameters"""
    # Basic control parameters
    velocity_scale: float = 0.1
    dt: float = 0.05
    damping: float = 1e-6
    
    # Joint limits and constraints
    theta_dot_max: float = 1.0
    theta_dot_min: float = -1.0
    joint_acceleration_limits: np.ndarray = None
    joint_jerk_limits: np.ndarray = None
    
    # Safety parameters
    max_force: float = 100.0  # Maximum allowable force in N
    collision_threshold: float = 0.01  # Minimum distance for collision detection
    emergency_stop_deceleration: float = 2.0  # Maximum deceleration during e-stop
    
    # Workspace limits
    workspace_limits: Dict[str, Tuple[float, float]] = None
    
    # Performance parameters
    control_frequency: float = 200  # Hz
    smoothing_factor: float = 0.1
    
    def __post_init__(self):
        """Initialize default values for None fields"""
        if self.workspace_limits is None:
            self.workspace_limits = {
                'x': (-0.5, 0.5),
                'y': (-0.5, 0.5),
                'z': (0.0, 0.6)
            }
        if self.joint_acceleration_limits is None:
            self.joint_acceleration_limits = np.ones(6) * 2.0
        if self.joint_jerk_limits is None:
            self.joint_jerk_limits = np.ones(6) * 5.0

class BaseController(ABC):
    """Advanced abstract base class for robot controllers"""
    
    def __init__(self, config: ControllerConfig):
        # Initialize configuration and state
        self.config = config
        self.state = ControllerState.INACTIVE
        
        # Initialize robot model components
        self.model = None
        self.data = None
        self.viz = None
        self.end_effector_frame = None
        
        # State variables
        self.q = None  # Joint positions
        self.dq = None  # Joint velocities
        self.ddq = None  # Joint accelerations
        
        # Initialize logging
        self.setup_logging()
        
        # Command queue for asynchronous execution
        self.command_queue = Queue()
        
        # Initialize robot and start monitoring
        self.setup_robot()
        self.start_monitoring()
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def setup_robot(self):
        """Initialize robot model and parameters"""
        try:
            # Load robot model
            self._load_robot_model()
            
            # Initialize state variables
            self.q = pin.neutral(self.model)
            self.dq = np.zeros(self.model.nv)
            self.ddq = np.zeros(self.model.nv)
            
            # Initialize trajectory buffers
            self.trajectory_buffer = []
            self.last_command_time = time.time()
            
            self.state = ControllerState.INITIALIZING
            self.logger.info("Robot setup completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error during robot setup: {e}")
            self.state = ControllerState.ERROR
            raise
        
    def _load_robot_model(self):
        """Load robot model from URDF"""
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        urdf_path = os.path.join(script_dir, "../urdf/new_Arm_Urdf.urdf")
        
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.end_effector_frame = self.model.getFrameId("Link_6")
        
    def start_monitoring(self):
        """Start monitoring threads"""
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
    def _monitor_loop(self):
        """Monitor robot state and safety"""
        while True:
            try:
                if self.state not in [ControllerState.INACTIVE, ControllerState.ERROR]:
                    self._check_safety()
                    self._process_commands()
                time.sleep(1.0 / self.config.control_frequency)
            except Exception as e:
                self.logger.error(f"Error in monitor loop: {e}")
                self.emergency_stop()
                
    def _check_safety(self):
        """Check various safety conditions"""
        try:
            # Check joint limits
            if not self._check_joint_limits():
                raise SafetyException("Joint limits exceeded")
            
            # Check velocities
            if not self._check_velocity_limits():
                raise SafetyException("Velocity limits exceeded")
            
            # Check workspace
            if not self._check_workspace_limits():
                raise SafetyException("Workspace limits exceeded")
            
            # Check collisions
            if self._check_collisions():
                raise SafetyException("Collision detected")
                
        except SafetyException as e:
            self.logger.error(f"Safety check failed: {e}")
            self.emergency_stop()
            
    def solve_ik(self, target_pose: np.ndarray, 
                 orientation: Optional[np.ndarray] = None, 
                 constraints: Optional[Dict] = None) -> Optional[np.ndarray]:
        """
        Enhanced inverse kinematics solver
        
        Args:
            target_pose: Target end-effector position
            orientation: Optional target orientation
            constraints: Optional dictionary of additional constraints
        """
        try:
            # Update forward kinematics
            pin.forwardKinematics(self.model, self.data, self.q)
            pin.updateFramePlacements(self.model, self.data)
            
            # Compute Jacobian
            J = pin.computeFrameJacobian(
                self.model, self.data, self.q,
                self.end_effector_frame,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
            )
            
            # Handle orientation constraints
            if orientation is not None:
                current_rotation = self.data.oMf[self.end_effector_frame].rotation
                orientation_error = self._compute_orientation_error(current_rotation, orientation)
                J = np.vstack([J[:3], J[3:]])  # Separate position and orientation Jacobian
                error = np.concatenate([target_pose - self.get_end_effector_position(), orientation_error])
            else:
                error = target_pose - self.get_end_effector_position()
                J = J[:3]  # Use only position Jacobian
                
            # Setup QP problem
            H = J.T @ J + self.config.damping * np.eye(self.model.nv)
            g = -J.T @ error
            
            # Add constraints
            C, b = self._setup_constraints(constraints)
            
            # Solve QP
            from quadprog import solve_qp
            dq = solve_qp(H, g, C.T, b)[0]
            
            # Apply smoothing
            if hasattr(self, 'last_dq'):
                dq = self.config.smoothing_factor * dq + (1 - self.config.smoothing_factor) * self.last_dq
            self.last_dq = dq
            
            return dq
            
        except Exception as e:
            self.logger.error(f"IK solver error: {e}")
            return None
            
    def _setup_constraints(self, constraints: Optional[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Setup constraint matrices for QP solver"""
        # Basic joint limits constraints
        C_basic = np.vstack([
            np.eye(self.model.nv),
            -np.eye(self.model.nv)
        ])
        b_basic = np.hstack([
            self.config.theta_dot_max * np.ones(self.model.nv),
            -self.config.theta_dot_min * np.ones(self.model.nv)
        ])
        
        if constraints is None:
            return C_basic, b_basic
            
        # Add additional constraints
        C_list = [C_basic]
        b_list = [b_basic]
        
        if 'acceleration_limits' in constraints:
            C_acc, b_acc = self._acceleration_constraints()
            C_list.append(C_acc)
            b_list.append(b_acc)
            
        if 'collision_avoidance' in constraints:
            C_col, b_col = self._collision_avoidance_constraints()
            C_list.append(C_col)
            b_list.append(b_col)
            
        return np.vstack(C_list), np.hstack(b_list)
    
    def get_end_effector_position(self) -> np.ndarray:
        """Get current end-effector position"""
        return self.data.oMf[self.end_effector_frame].translation
        
    def get_end_effector_orientation(self) -> np.ndarray:
        """Get current end-effector orientation"""
        return self.data.oMf[self.end_effector_frame].rotation
        
    def emergency_stop(self):
        """Enhanced emergency stop procedure"""
        self.state = ControllerState.EMERGENCY_STOP
        self.logger.warning("Emergency stop activated")
        
        # Calculate safe deceleration
        max_decel = self.config.emergency_stop_deceleration
        stop_time = np.max(np.abs(self.dq)) / max_decel
        
        # Generate deceleration trajectory
        t = 0
        dt = self.config.dt
        while t < stop_time:
            decel_factor = 1 - (t / stop_time)
            self.dq *= decel_factor
            self.q = pin.integrate(self.model, self.q, self.dq * dt)
            t += dt
            
        # Final stop
        self.dq = np.zeros(self.model.nv)
        self.ddq = np.zeros(self.model.nv)
        
    @abstractmethod
    def compute_control(self) -> Optional[np.ndarray]:
        """
        Abstract method to compute control command
        Must be implemented by derived controllers
        """
        pass
    
    def add_command(self, command: Dict[str, Any]):
        """Add command to execution queue"""
        self.command_queue.put(command)
        
    def _process_commands(self):
        """Process commands from queue"""
        while not self.command_queue.empty():
            command = self.command_queue.get()
            try:
                if command['type'] == 'move':
                    self._execute_move_command(command)
                elif command['type'] == 'trajectory':
                    self._execute_trajectory_command(command)
                # Add more command types as needed
            except Exception as e:
                self.logger.error(f"Error processing command: {e}")
                
    def _execute_move_command(self, command):
        """Execute a move command"""
        target_pose = command.get('target_pose')
        orientation = command.get('orientation')
        constraints = command.get('constraints')
        
        dq = self.solve_ik(target_pose, orientation, constraints)
        if dq is not None:
            self.update_configuration(dq)
            
    def update_configuration(self, dq: np.ndarray):
        """Update robot configuration with safety checks"""
        try:
            # Store previous state
            q_prev = self.q.copy()
            dq_prev = self.dq.copy()
            
            # Update state
            self.dq = dq
            self.ddq = (self.dq - dq_prev) / self.config.dt
            self.q = pin.integrate(self.model, self.q, self.dq * self.config.dt)
            
            # Verify update is safe
            if self._verify_state_update():
                # Update visualization if available
                if hasattr(self, 'viz'):
                    self.viz.display(self.q)
            else:
                # Revert to previous state if update is unsafe
                self.q = q_prev
                self.dq = dq_prev
                self.logger.warning("Configuration update reverted due to safety checks")
                
        except Exception as e:
            self.logger.error(f"Error updating configuration: {e}")
            self.emergency_stop()
            
    def _verify_state_update(self) -> bool:
        """Verify that state update is safe"""
        return (self._check_joint_limits() and 
                self._check_velocity_limits() and 
                self._check_workspace_limits() and 
                not self._check_collisions())

class SafetyException(Exception):
    """Custom exception for safety-related issues"""
    pass

class VelocityIKController(BaseController):
    """Controller for velocity-based inverse kinematics"""

    def __init__(self, config: ControllerConfig, model: pin.Model, data: pin.Data, viz: Optional[pin.visualize.MeshcatVisualizer] = None):
        super().__init__(config, model, data, viz)
        
        