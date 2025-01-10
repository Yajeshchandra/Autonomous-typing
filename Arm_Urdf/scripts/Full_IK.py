#!/usr/bin/env python3

import pinocchio as pin
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtCore import Qt, QTimer
from quadprog import solve_qp  # Install with `pip install quadprog`
import os
from pinocchio.visualize import MeshcatVisualizer

# Load the robot model
urdf_path = os.path.join("Arm_Urdf", "urdf", "Arm_Urdf.urdf")  # Replace with the path to your URDF file
model = pin.buildModelFromUrdf(urdf_path)
data = model.createData()

# Visualization setup
visual_model = pin.buildGeomFromUrdf(model, urdf_path, pin.GeometryType.VISUAL)
collision_model = pin.buildGeomFromUrdf(model, urdf_path, pin.GeometryType.COLLISION)
viz = MeshcatVisualizer(model, collision_model, visual_model)
viz.initViewer(loadModel=True)

# End-effector frame
end_effector_frame = model.getFrameId("Gripper_2")  # Replace with your end-effector frame name

# Initialize joint configuration
q = pin.neutral(model)
viz.display(q)

# Velocity scaling
velocity_scale = 0.1  # Adjust this for desired velocity magnitude
dt = 0.05  # Time step for integration
damping = 1e-6  # Regularization factor

# Joint limits
q_min = model.lowerPositionLimit
q_max = model.upperPositionLimit

# Key-to-twist mapping
key_twist_mapping = {
    Qt.Key_W: np.array([velocity_scale, 0, 0, 0, 0, 0]),  # Forward
    Qt.Key_S: np.array([-velocity_scale, 0, 0, 0, 0, 0]), # Backward
    Qt.Key_A: np.array([0, velocity_scale, 0, 0, 0, 0]), # Left
    Qt.Key_D: np.array([0, -velocity_scale, 0, 0, 0, 0]), # Right
    Qt.Key_Q: np.array([0, 0, velocity_scale, 0, 0, 0]), # Up
    Qt.Key_E: np.array([0, 0, -velocity_scale, 0, 0, 0]), # Down
    Qt.Key_J: np.array([0, 0, 0, -velocity_scale, 0, 0]), # Rotate around x towards left
    Qt.Key_L: np.array([0, 0, 0, velocity_scale, 0, 0]), # Rotate around x towards right
    Qt.Key_I: np.array([0, 0, 0, 0, velocity_scale, 0]), # Rotate around y down
    Qt.Key_K: np.array([0, 0, 0, 0, -velocity_scale, 0]), # Rotate around y up
    Qt.Key_U: np.array([0, 0, 0, 0, 0, velocity_scale]), # Yaw left
    Qt.Key_O: np.array([0, 0, 0, 0, 0, -velocity_scale]), # Yaw right
}

class VelocityIKController(QWidget):
    def __init__(self):
        super().__init__()
        self.pressed_keys = set()
        self.timer = QTimer()
        self.timer.timeout.connect(self.control_loop)
        self.timer.start(int(dt * 1000))

    def keyPressEvent(self, event):
        self.pressed_keys.add(event.key())

    def keyReleaseEvent(self, event):
        self.pressed_keys.discard(event.key())

    def compute_desired_twist(self):
        desired_twist = np.zeros(6)
        for key in self.pressed_keys:
            if key in key_twist_mapping:
                desired_twist += key_twist_mapping[key]
        return desired_twist

    def control_loop(self):
        global q

        # Compute forward kinematics and Jacobian
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        pin.computeJointJacobians(model, data, q)

        J = pin.computeFrameJacobian(model, data, q, end_effector_frame, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)

        # Get the desired twist from key input
        desired_twist = self.compute_desired_twist()

        if np.linalg.norm(desired_twist) > 1e-6:  # If there is a desired motion
            # Quadratic program matrices
            H = J.T @ J + damping * np.eye(model.nv)  # Regularized Hessian
            g = -J.T @ desired_twist  # Gradient term

            # Inequality constraints for joint velocity limits
            theta_dot_max = 1.0 * np.ones(model.nv)
            theta_dot_min = -1.0 * np.ones(model.nv)

            # Constraints for joint position limits
            q_upper_violation = (q_max - q) / dt
            q_lower_violation = (q_min - q) / dt

            # Combine velocity and position constraints
            C = np.vstack([np.eye(model.nv), -np.eye(model.nv), np.eye(model.nv), -np.eye(model.nv)])
            b = np.hstack([theta_dot_min, -theta_dot_max, q_lower_violation, -q_upper_violation])

            # Solve the quadratic program
            theta_dot = solve_qp(H, g, C.T, b)[0]

            # Update joint configuration using integration
            q = pin.integrate(model, q, theta_dot * dt)
            viz.display(q)

if __name__ == "__main__":
    app = QApplication([])
    controller = VelocityIKController()
    controller.show()
    app.exec_()
