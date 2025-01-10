import pinocchio as pin
import numpy as np
from quadprog import solve_qp  # Install with `pip install quadprog`
import time
import os
# Load the robot model
# Get the absolute path to the directory containing this script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Build the full path to the URDF file relative to the script's location
urdf_path = os.path.join(script_dir, "../urdf/Arm_Urdf.urdf")
model = pin.buildModelFromUrdf(urdf_path)
visual_model = pin.buildGeomFromUrdf(model, urdf_path, pin.GeometryType.VISUAL)
collision_model = pin.buildGeomFromUrdf(model, urdf_path, pin.GeometryType.COLLISION)
data = model.createData()

# Initialize the visualizer
viz = pin.visualize.MeshcatVisualizer(model, collision_model, visual_model)
viz.initViewer(loadModel=True)
# Desired end-effector pose (as SE(3))
desired_pose = pin.SE3(np.eye(3), np.array([0.5, 0.2, 0.3]))  # Replace with your target pose

# Initialize joint configuration
q = pin.neutral(model)  # Starting at the neutral configuration
max_iterations = 100
tolerance = 1e-4
damping = 1e-6  # Regularization factor for stability

# Joint limits
q_min = model.lowerPositionLimit  # Replace with your robot's joint limits
q_max = model.upperPositionLimit   # Replace with your robot's joint limits

# End-effector frame index (modify based on your URDF)
end_effector_frame = model.getFrameId("Link_6")  # Replace with your end-effector frame name
viz.display(q)
time.sleep(5)
# Solver loop
for i in range(max_iterations):
    # Compute forward kinematics and Jacobian
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    pin.computeJointJacobians(model, data, q)

    # Current end-effector pose
    current_pose = data.oMf[end_effector_frame]

    # Error in SE(3) as a 6D vector (translation + rotation)
    error = pin.log6(desired_pose.inverse() * current_pose).vector
    if np.linalg.norm(error) < tolerance:
        print(f"Converged in {i+1} iterations")
        break

    # Compute Jacobian in local frame since the error is in the end-effector frame
    J = pin.computeFrameJacobian(model, data, q, end_effector_frame, pin.ReferenceFrame.LOCAL)

    # Quadratic program matrices
    H = J.T @ J + damping * np.eye(model.nq)  # Regularized Hessian
    g = -J.T @ error  # Gradient term

    # Inequality constraints for joint limits
    # Ensuring q_min <= q + Δq <= q_max
    C = np.vstack([np.eye(model.nq), -np.eye(model.nq)])
    b = np.hstack([q_min - q, -(q_max - q)])

    # Solve QP
    delta_q = solve_qp(H, g, C.T, b)[0]  # Solving the QP with constraints

    # Update joint configuration
    q = pin.integrate(model, q, delta_q)
    viz.display(q)
    time.sleep(0.2)

# Print final joint configuration
print("Final joint configuration:", q, "\n", "Current pose:", data.oMf[end_effector_frame])
