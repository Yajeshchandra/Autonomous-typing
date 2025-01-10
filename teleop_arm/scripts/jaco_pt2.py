#!/usr/bin/env python3

import rospy
import numpy as np
import math  # To capture keyboard input
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
import sys, select, termios, tty

# Define key mappings for velocities
key_mapping = {
    'w': (0.02, 0, 0),  # Set X velocity
    'x': (-0.02, 0, 0),     # Stop X velocity
    'a': (0, 0.02, 0),  # Set Y velocity
    'd': (0, -0.02, 0), # Set Y velocity
    'e': (0, 0, 0.02),  # Set Z velocity
    'z': (0, 0, -0.02)  # Set Z velocity
}

joint_angles = np.zeros(3)

# Get keyboard input in the terminal
def get_key():
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ''
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

def Rz(x):
    return np.array([[math.cos(x), -math.sin(x), 0],
                     [math.sin(x),  math.cos(x), 0],
                     [0, 0, 1]])

def Ry(x):    
    return np.array([[math.cos(x), 0, math.sin(x)],
                     [0, 1, 0],
                     [-math.sin(x), 0, math.cos(x)]])

# Function to compute transformation matrix and Jacobian inverse
def get_matrix(joint_angles):
    A = 0.1745
    B = 0.435
    C = 0.335 + 0.060
    a,b,c = joint_angles
    # Homogeneous transformations
    R_0_1 = np.dot(Rz(a), np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]))
    d_0_1 = np.array([0, 0, A])
    
    R_1_2 = np.dot(Rz(b), np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]]))
    d_1_2 = np.array([B * math.sin(b), -B * math.cos(b), 0])
    
    R_2_3 = np.dot(Rz(c), np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]))
    d_2_3 = np.array([C * math.sin(c), -C * math.cos(c), 0])
    
    R_0_2 = R_0_1 @ R_1_2
    R_0_3 = R_0_2 @ R_2_3
    
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
    
    return jaco_inv

# Function to compute joint velocities using the inverse Jacobian
def compute_joint_velocities(X, Y, Z, joint_angles):
    J = get_matrix(joint_angles)
    end_effector_velocities = np.array([X, Y, Z])
    joint_velocities = J.dot(end_effector_velocities)
    return joint_velocities

# Function to publish joint velocities
def publish_joint_velocities(pub, joint_velocities):
    traj = JointTrajectory()
    traj.joint_names = ['joint_0', 'joint_1', 'joint_2', 'joint_3']

    msg = Float64MultiArray()

    # point = JointTrajectoryPoint()
    # joint_angles += joint_velocities  # Update joint angles based on joint velocities
    # point.positions = [joint_angles[0], joint_angles[1], joint_angles[2], 0]  # Assuming joint_3 is fixed at 0 for simplicity
    # point.time_from_start = rospy.Duration(1)
    # traj.points = [point]

    #joint_angles += joint_velocities
    angles = joint_angles + joint_velocities*0.1
    msg.data = [angles[0], angles[1], angles[2], 0,0]

    rospy.loginfo(f"Publishing joint velocities: {joint_velocities}")
    pub.publish(msg)

def joint_sub(msg : JointState):
    global joint_angles
    joint_angles = np.array(msg.position[:3])

def main():
    global joint_angles
    rospy.init_node('teleop_arm')
    pub = rospy.Publisher('/plan_controller/command', Float64MultiArray, queue_size=10)
    sub = rospy.Subscriber('/joint_states', JointState, joint_sub)
    rate = rospy.Rate(1000)  # 10 Hz

    X, Y, Z = 0.0, 0.0, 0.0  # Initial end-effector velocities
    # joint_angles = np.array([0.0, 0.0, 0.0])  # Initial joint angles

    rospy.loginfo("Teleoperation node started. Use keys to control the arm.")

    while not rospy.is_shutdown():
        
        # Publish the joint velocities
        key = get_key()

        # Update end-effector velocities based on key inputs
        if key in key_mapping:
            dX, dY, dZ = key_mapping[key]
            X, Y, Z = dX, dY, dZ  # Set velocities directly based on the key pressed

            rospy.loginfo(f"Updated velocities: X={X}, Y={Y}, Z={Z}")

            # Compute joint velocities from the end-effector velocities
            joint_velocities = compute_joint_velocities(X, Y, Z, joint_angles)

            # Publish the joint velocities
            publish_joint_velocities(pub, joint_velocities)

        else:
            # Reset velocities if no valid key is pressed
            X, Y, Z = 0.0, 0.0, 0.0
            joint_velocities = compute_joint_velocities(0, 0, 0, joint_angles)
            publish_joint_velocities(pub, joint_velocities)

        if key == '\x03':  # Ctrl+C to stop
            rospy.loginfo("Shutting down teleoperation node.")
            break

        rate.sleep()

if __name__ == '__main__':
    settings = termios.tcgetattr(sys.stdin)
    try:
        main()
    except rospy.ROSInterruptException:
        pass
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
