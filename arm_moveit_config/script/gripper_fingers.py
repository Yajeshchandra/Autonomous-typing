#!/usr/bin/env python3

import rospy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import sys, select, termios, tty

class TeleopGripperController:
    def __init__(self):
        # Define the joint names for the gripper
        self.joint_names = ['Finger_1', 'Finger_2']
        # Initial joint angles for open position
        self.joint_angles = [0.0, 0.0]  # Finger_1 at 0.0, Finger_2 at 0.0
        self.step_size = 0.001  # Incremental step for each key press
        self.settings = termios.tcgetattr(sys.stdin)

        rospy.init_node('teleop_gripper')
        self.pub = rospy.Publisher('/end_effector_controller/command', JointTrajectory, queue_size=10)
        self.rate = rospy.Rate(10)  # 10 Hz

        rospy.loginfo("Teleoperation node started. Use 'o' to incrementally open and 'c' to incrementally close the gripper.")

    def get_key(self):
        tty.setraw(sys.stdin.fileno())
        select.select([sys.stdin], [], [], 0)
        key = sys.stdin.read(1)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        return key

    def run(self):
        while not rospy.is_shutdown():
            key = self.get_key()
            command_executed = False

            if key == 'o':  # Incrementally open gripper
                self.joint_angles[0] = max(0.0, self.joint_angles[0] - self.step_size)
                self.joint_angles[1] = min(0.0, self.joint_angles[1] + self.step_size)
                command_executed = True
                rospy.loginfo("Gripper opening incrementally.")

            elif key == 'c':  # Incrementally close gripper
                self.joint_angles[0] = min(0.0354, self.joint_angles[0] + self.step_size)
                self.joint_angles[1] = max(-0.0354, self.joint_angles[1] - self.step_size)
                command_executed = True
                rospy.loginfo("Gripper closing incrementally.")

            elif key == '\x03':  # Ctrl+C
                rospy.loginfo("Shutting down teleoperation node.")
                break

            if command_executed:
                # Create the JointTrajectory message
                trajectory_msg = JointTrajectory()
                trajectory_msg.joint_names = self.joint_names

                # Create a single point in the trajectory
                point = JointTrajectoryPoint()
                point.positions = self.joint_angles
                point.time_from_start = rospy.Duration(0.1)  # Small duration to indicate immediate movement

                trajectory_msg.points = [point]

                # Publish the trajectory message
                self.pub.publish(trajectory_msg)

                rospy.loginfo(f"Published joint angles: {self.joint_angles}")

            self.rate.sleep()

    def shutdown(self):
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)


if __name__ == '__main__':
    teleop_gripper = TeleopGripperController()
    try:
        teleop_gripper.run()
    except rospy.ROSInterruptException:
        pass
    finally:
        teleop_gripper.shutdown()
