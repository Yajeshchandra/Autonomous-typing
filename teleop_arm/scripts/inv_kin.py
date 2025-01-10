#!/usr/bin/env python3

import rospy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
import sys, select, termios, tty
import math

class JointAngleReader:
    def __init__(self):
        self.joint_angles = {}
        # rospy.init_node('joint_angle_reader', anonymous=True)
        rospy.Subscriber("/joint_states", JointState, self.joint_state_callback)

    def joint_state_callback(self, msg):
        # msg.position contains the angles of the joints

        joint_names = msg.name
        joint_positions = msg.position
        self.joint_angles = dict(zip(joint_names, joint_positions))

        # rospy.loginfo(f"Current Joint Angles: {self.joint_angles}")
        # for joint, angle in self.joint_angles.items():
        #     rospy.loginfo(f"Joint: {joint}, Angle: {angle:.3f}")
        
    def get_joint_angles(self):
        return self.joint_angles

class TeleopArmController:
    def __init__(self):
        # Define the keys and their corresponding joint indices
        self.joint_names = ['joint_0', 'joint_1', 'joint_2', 'joint_3', 'joint_4']
        # Initial joint angles
        self.joint_angles = [0.0] * len(self.joint_names)
        self.joint_step = 0.01  # Step size for changing joint angles

        self.dist = 5*1e-5 # for joint 1 and joint


        # Initialize the selected joint
        self.selected_joint = 0
        self.l1 = 0.510  # Length of link 1
        self.l2 = 0.360 # Length of link 2

        joint_angle_reader = JointAngleReader()

        rospy.loginfo("Waiting for joint states to be available...")
        rospy.sleep(1.0)  # Give it a second to receive joint data

        joint_angles_dict = joint_angle_reader.get_joint_angles() 
        rospy.loginfo(f"Current Joint Angles: {joint_angles_dict}")


        # theta1,theta2 = joint_angles_dict['joint_1'],joint_angles_dict['joint_2']

        self.current_x = self.l2  # Initial x position
        self.current_y = self.l1 # Initial y position (always 0 for forward motion)

        # self.current_x,self.current_y = self.fwd_kinematics(theta1,theta2)

        self.settings = termios.tcgetattr(sys.stdin)

        rospy.init_node('teleop_arm')
        self.pub = rospy.Publisher('/robot_arm_controller/command', JointTrajectory, queue_size=10)
        self.rate = rospy.Rate(10)  # 10 Hz

        rospy.loginfo("Teleoperation node started. Select joint (0,3 or 4) and adjust using 'w' (increase) or 's' (decrease). Press '9' to move joint 2 and joint 3 forward simultaneously and press '8' for backward movement")

    def get_key(self):
        tty.setraw(sys.stdin.fileno())
        select.select([sys.stdin], [], [], 0)
        key = sys.stdin.read(1)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        return key
    
    def fwd_kinematics(self,theta1,theta2):
        # Given link lengths and joint angles (in radians)
        l1 = self.l1 # Length of link 1
        l2 = self.l2


        # Calculate the end effector position (x, y)
        x = l1 * math.sin(theta1) + l2 * math.cos(theta2)
        y = l1 * math.cos(theta1) + l2 * math.sin(theta2)

        # print(f"x: {x}")
        # print(f"y: {y}")
        
        return theta1,theta2

    
    def inverse_kinematics(self, x, y):

        # theta2 = math.acos((x**2 + y**2 - self.l1**2 - self.l2**2) / (2 * self.l1 * self.l2))

        cos_theta2 = (x**2 + y**2 - self.l1**2 - self.l2**2) / (2 * self.l1 * self.l2)
        cos_theta2 = max(min(cos_theta2, 1), -1)  # Clamping the value between -1 and 1
        theta2 = math.acos(cos_theta2)


        # theta1a = math.atan2(y, x) - math.atan2(self.l2 * math.sin(theta2), self.l1 + self.l2 * math.cos(theta2))

        theta1b = math.atan2(y, x) + math.atan2(self.l2 * math.sin(theta2), self.l1 + self.l2 * math.cos(theta2))


        theta1 = theta1b

        theta1_adj = math.pi/2 - theta1
        theta2_adj = math.pi/2 - theta2 

        rospy.loginfo(f"Theta1: {theta1_adj}")
        rospy.loginfo(f"Theta2: {theta2_adj}")
    
        return theta1_adj, theta2_adj

    

    def move_fwd(self):
        start_x = self.current_x
        distance = self.dist # change this accordingly
        end_x = start_x + distance
        
        
        # current_x = start_x + distance
        
        theta1, theta2 = self.inverse_kinematics(end_x, self.current_y)
        
        
        self.joint_angles[1] += theta1
        self.joint_angles[1] = max(-3.14, min(3.14, self.joint_angles[1]))
        
        self.joint_angles[2] += theta2
        self.joint_angles[2] = max(-3.14, min(3.14, self.joint_angles[2]))

        
        self.current_x = end_x
        return 
    
    def move_back(self):
        start_x = self.current_x
        distance = self.dist
        end_x = start_x + distance
        
        
        
        theta1, theta2 = self.inverse_kinematics(end_x, self.current_y)
        
        
        self.joint_angles[1] -= theta1
        self.joint_angles[1] = max(-3.14, min(3.14, self.joint_angles[1]))
        
        self.joint_angles[2] -= theta2
        self.joint_angles[2] = max(-3.14, min(3.14, self.joint_angles[2]))

        
        self.current_x = end_x
        return 

    def run(self):
        while not rospy.is_shutdown():
            key = self.get_key()
            command_executed = False
            kinematics = False

            if key in '034':
                self.selected_joint = int(key)
                rospy.loginfo(f"Joint {self.selected_joint} selected.")

            if key == '9':
                self.move_fwd()
                rospy.loginfo(f"Joint 1 and joint 2 selected.")
                command_executed = True
                kinematics = True

            if key == '8':
                self.move_back()
                rospy.loginfo(f"Joint 1 and joint 2 selected.")
                command_executed = True
                kinematics = True


            elif key == 'w':  # Increase joint angle
                self.joint_angles[self.selected_joint] += self.joint_step
                command_executed = True

            elif key == 's':  # Decrease joint angle
                self.joint_angles[self.selected_joint] -= self.joint_step
                command_executed = True

            elif key == '\x03':  # Ctrl+C
                rospy.loginfo("Shutting down teleoperation node.")
                break

            if command_executed:
                # Clamp the angles within a range if needed
                if not kinematics :
                    self.joint_angles[self.selected_joint] = max(-3.14, min(3.14, self.joint_angles[self.selected_joint]))
                    rospy.loginfo(f"Adjusted joint {self.selected_joint} to {self.joint_angles[self.selected_joint]} radians.")

                # Create the JointTrajectory message
                trajectory_msg = JointTrajectory()
                trajectory_msg.joint_names = self.joint_names

                # Create a single point in the trajectory
                point = JointTrajectoryPoint()
                point.positions = self.joint_angles
                point.time_from_start = rospy.Duration(0.01)  # Small duration to indicate immediate movement

                trajectory_msg.points = [point]

                # Publish the trajectory message
                self.pub.publish(trajectory_msg)

                rospy.loginfo(f"Published joint angles: {self.joint_angles}")

            self.rate.sleep()

    def shutdown(self):
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)


if __name__ == '__main__':
    teleop_arm = TeleopArmController()
    try:
        teleop_arm.run()
    except rospy.ROSInterruptException:
        pass
    finally:
        teleop_arm.shutdown()