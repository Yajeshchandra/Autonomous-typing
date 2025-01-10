#!/usr/bin/env python3

import rospy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import math

def inv_kin(pub, a, b, c, d=0):
    # Create JointTrajectory message
    traj = JointTrajectory()
    traj.joint_names = ['joint_0', 'joint_1', 'joint_2', 'joint_3']
    
    # Create JointTrajectoryPoint
    point = JointTrajectoryPoint()
    point.positions = [a, b, c, d]  # Set joint angles
    point.time_from_start = rospy.Duration(1)  # Time to reach the target position
    
    traj.points = [point]

    # Log and publish joint angles
    rospy.loginfo(f"Publishing joint angles: {a}, {b}, {c}, {d}")
    pub.publish(traj)

def clamp(value, min_val, max_val):
    """Clamps a value to stay within the specified range"""
    return max(min(value, max_val), min_val)

def calculate_joint_angles(x, y, z):
    print(f"Received inputs - x: {x}, y: {y}, z: {z}")
    
    # Robot-specific constants
    a1 = 0.1745  # Base height
    a2 = 0.435   # Link length 1
    a3 = 0.335+0.060 # Link length 2
    
    # Joint 0: Base rotation (arctangent of y/x)
    a = math.atan2(y, x)
    print("joint_0 (base rotation):", a)
    
    # r is the distance from the base to the projection of (x, y) on the xy-plane
    r = (math.pow(abs(z - a1), 2) + math.pow(x * math.cos(a), 2)) ** 0.5
    print("r (distance to projection on xy-plane):", r)
    
    # Clamp values for fi calculation to avoid domain errors
    fi_arg = ((r ** 2) + (a3 ** 2) - (a2 ** 2)) / (2 * r * a3)
    # fi_arg = clamp(fi_arg, -1.0, 1.0)  # Ensure the argument stays in [-1, 1]
    fi = math.acos(fi_arg)
    print("fi:", fi)
    
    # Joint 1: Shoulder angle
    b = 1.57-math.atan2(abs(z - a1), x * math.cos(a)) - fi
    print("joint_1 (shoulder angle):", b)
    
    # Joint 2: Elbow angle
    c_arg = (abs(z - a1) - (a2 * math.cos(b))) / a3
    # c_arg = clamp(c_arg, -1.0, 1.0)  # Ensure the argument stays in [-1, 1]
    c = math.asin(c_arg)
    print("joint_2 (elbow angle):", c)
    
    print(f"Calculated joint angles - a: {a}, b: {b}, c: {c}")
    return a, b, c

def main():
    rospy.init_node('inverse_kin', anonymous=True)
    
    # Initialize publisher outside the loop
    pub = rospy.Publisher('/plan_controller/command', JointTrajectory, queue_size=10)
    
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        try:
            # Take user input for x, y, z
            x = float(input("Enter x-coordinate: "))
            y = float(input("Enter y-coordinate: "))
            z = float(input("Enter z-coordinate: "))

            # Calculate joint angles based on x, y, z
            a, b, c = calculate_joint_angles(x, y, z)

            # Publish the angles
            inv_kin(pub, a, b, c)

            rate.sleep()

        except rospy.ROSInterruptException:
            break
        except ValueError as e:
            print(f"Invalid input. Please enter numeric values. Error: {e}")
        except Exception as e:
            print(f"Unexpected error occurred: {e}")

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
