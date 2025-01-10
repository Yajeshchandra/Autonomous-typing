#!/usr/bin/env python3

import rospy
import numpy as np
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import math

# Rotation matrices
def Rz(x):
    return np.array([[math.cos(x), -math.sin(x), 0],
                     [math.sin(x),  math.cos(x), 0],
                     [0, 0, 1]])

def Ry(x):    
    return np.array([[math.cos(x), 0, math.sin(x)],
                     [0, 1, 0],
                     [-math.sin(x), 0, math.cos(x)]])

# Function to compute transformation matrix and Jacobian inverse
def get_matrix(a, b, c):
    A = 0.1745
    B = 0.435
    C = 0.335 + 0.060

    # Homogeneous transformations
    R_0_1 = np.dot(Rz(a), np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]))
    d_0_1 = np.array([0, 0, A])
    
    R_1_2 = np.dot(Rz(b), np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]]))
    d_1_2 = np.array([B * math.sin(b), -B * math.cos(b), 0])
    
    R_2_3 = np.dot(Rz(c), np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]))
    d_2_3 = np.array([C * math.sin(c), -C * math.cos(c), 0])
    
    R_0_2 = R_0_1 @ R_1_2
    R_0_3 = R_0_2 @ R_2_3
    
    # Homogeneous transformations
    H_0_1 = np.vstack((np.hstack((R_0_1, d_0_1.reshape(3, 1))), np.array([0, 0, 0, 1])))
    H_1_2 = np.vstack((np.hstack((R_1_2, d_1_2.reshape(3, 1))), np.array([0, 0, 0, 1])))
    H_2_3 = np.vstack((np.hstack((R_2_3, d_2_3.reshape(3, 1))), np.array([0, 0, 0, 1])))

    H_0_2 = H_0_1 @ H_1_2
    H_0_3 = H_0_1 @ H_1_2 @ H_2_3
    
    d_0_2 = H_0_2[:3, 3]
    d_0_3 = H_0_3[:3, 3]
    
    # Jacobian columns (cross products)
    col1 = np.cross([0, 0, 1], d_0_3)
    col2 = np.cross(R_0_1 @ [0, 0, 1], d_0_3 - d_0_1)
    col3 = np.cross(R_0_2 @ [0, 0, 1], d_0_3 - d_0_2)

    # Construct Jacobian
    jaco = np.column_stack((col1, col2, col3))
    
    # Invert the Jacobian (use pseudo-inverse to avoid singularities)
    jaco_inv = np.linalg.inv(jaco)
    
    print(np.linalg.matrix_rank(jaco_inv))
    return jaco_inv

# Function to publish joint velocities
def inv_kin(pub, joint_angles):
    traj = JointTrajectory()
    traj.joint_names = ['joint_0', 'joint_1', 'joint_2', 'joint_3']
    
    point = JointTrajectoryPoint()
    point.positions = [joint_angles[0], joint_angles[1], joint_angles[2], 0]  # Set joint angles
    point.time_from_start = rospy.Duration(1)
    traj.points = [point]

    rospy.loginfo(f"Publishing joint angles: {point.positions}")
    pub.publish(traj)

def main():
    # Initialize the ROS node
    rospy.init_node('matrix_calculator_and_velocity_publisher', anonymous=True)
    pub = rospy.Publisher('/plan_controller/command', JointTrajectory, queue_size=10)

    rate = rospy.Rate(5)  # 2 Hz loop rate
    
    # Initial joint angles
    a, b, c = 0, 0, 0

    while not rospy.is_shutdown():
        # Get the Jacobian inverse matrix based on the joint angles
        matrix = get_matrix(a, b, c)

        try:
            # Set desired velocities (for demonstration, set static or predefined values)
            X, Y, Z = 0.0, 0.0, 0.01  # Example static velocities for testing
            vector_3x1 = np.array([X, Y, Z]).reshape(3, 1)
            joint_velocities = np.dot(matrix, vector_3x1).flatten()
            # Multiply inverse Jacobian by the velocity vector to calculate joint velocities
            # alpha = 0.8  # Smoothing factor (between 0 and 1)
            # prev_joint_velocities = np.zeros(3)  # Initial velocities

            # # Smooth velocities
            # joint_velocities = alpha * joint_velocities + (1 - alpha) * prev_joint_velocities
            # prev_joint_velocities = joint_velocities  # Update for the next iteration

            

            # a += joint_velocities[0]
            # b += joint_velocities[1]
            # c += joint_velocities[2]
            
              # Assuming 0 for the fourth joint
            
            thresh=1.5
            if -thresh<joint_velocities[0] < thresh:
                a += joint_velocities[0]
            else:
                a+=0
                rospy.logwarn("Joint 0 angle exceeded limits, stopping increment.")

            if -thresh<joint_velocities[1] < thresh:
                b+=0
                b += joint_velocities[1]
            else:
                rospy.logwarn("Joint 1 angle exceeded limits, stopping increment.")

            if -thresh<joint_velocities[2] < thresh:
                c+=0
                c += joint_velocities[2]
            else:
                rospy.logwarn("Joint 2 angle exceeded limits, stopping increment.")

            joint_angles = [a, b, c, 0.0]
                
            # Publish the joint angles
            inv_kin(pub, joint_angles)

            # Log the updated joint angles
            rospy.loginfo(f"Updated joint angles: {joint_angles}")
        
        except np.linalg.LinAlgError:
            rospy.logerr("The Jacobian matrix is singular and cannot be inverted.")
        
        rate.sleep()

if __name__ == "__main__":
    main()
