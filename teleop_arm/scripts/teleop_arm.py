#!/usr/bin/env python3

import rospy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import sys, select, termios, tty
from std_msgs.msg import Float64MultiArray

# Define the keys and their corresponding joint indices
joint_names = ['joint_0', 'joint_1', 'joint_2', 'joint_3']
key_mapping = {
    'a': 0, 'z': 0,  # Increase/decrease joint_0
    's': 1, 'x': 1,  # Increase/decrease joint_1
    'd': 2, 'c': 2,  # Increase/decrease joint_2
    'f': 3, 'v': 3,  # Increase/decrease joint_3
}

# Initial joint angles
joint_angles = [0.0] * len(joint_names)
joint_step = 0.1  # Step size for changing joint angles

def get_key():
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ''
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


def main():
    global joint_angles
    rospy.init_node('teleop_arm')
    pub = rospy.Publisher('/plan_controller/command', Float64MultiArray, queue_size=10)
    rate = rospy.Rate(10)  # 10 Hz

    rospy.loginfo("Teleoperation node started. Use keys to control the arm.")

    while not rospy.is_shutdown():
        keys = ''
        while True:
            key = get_key()
            if key == '':
                break
            keys += key

        command_executed = False

        for key in keys:
            if key in key_mapping:
                joint_index = key_mapping[key]
                if key in 'asdf':  # Increase joint angle
                    joint_angles[joint_index] += joint_step
                elif key in 'zxcv':  # Decrease joint angle
                    joint_angles[joint_index] -= joint_step

                # Clamp the angles within a range if needed
                joint_angles[joint_index] = max(-3.14, min(3.14, joint_angles[joint_index]))

                command_executed = True
                rospy.loginfo(f"Key '{key}' pressed. Adjusting joint {joint_index} to {joint_angles[joint_index]} radians.")

        if '\x03' in keys:  # Ctrl+C
            rospy.loginfo("Shutting down teleoperation node.")
            break

        # Create the JointTrajectory message
        trajectory_msg = JointTrajectory()
        trajectory_msg.joint_names = joint_names
        
        # Create a single point in the trajectory
        point = JointTrajectoryPoint()
        point.positions = joint_angles
        point.time_from_start = rospy.Duration(0.1)  # Small duration to indicate immediate movement
        
        trajectory_msg.points = [point]

        # Publish the trajectory message

        msg = Float64MultiArray()
        msg.data = joint_angles
        pub.publish(msg)
        
        if command_executed:
            rospy.loginfo(f"Published joint angles: {joint_angles}")

        rate.sleep()

if __name__ == '__main__':
    settings = termios.tcgetattr(sys.stdin)
    try:
        main()
    except rospy.ROSInterruptException:
        pass
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
