#!/usr/bin/env python3


import sys
import rospy
import moveit_commander
from geometry_msgs.msg import Pose

def main():
    # Initialize the moveit_commander and ROS node
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('move_arm', anonymous=True)

    # Instantiate a RobotCommander object to interact with the robot as a whole
    robot = moveit_commander.RobotCommander()

    # Instantiate a PlanningSceneInterface object to interact with the world around the robot
    scene = moveit_commander.PlanningSceneInterface()

    # Instantiate a MoveGroupCommander object for the arm
    group_name = "xarm7"  # Replace 'arm' with the name of your move group
    move_group = moveit_commander.MoveGroupCommander(group_name)

    # Get the current pose of the end effector
    current_pose = move_group.get_current_pose().pose
    rospy.loginfo(f"Current Pose: {current_pose}")

    # Define a new target pose relative to the current pose
    target_pose = Pose()
    target_pose.position.x = current_pose.position.x  # Move 0.1 meters in x direction
    target_pose.position.y = current_pose.position.y  # Move 0.1 meters in y direction
    target_pose.position.z = current_pose.position.z + 0.1  # Move 0.1 meters in z direction
    target_pose.orientation = current_pose.orientation  # Keep the same orientation

    # Set the target pose for the MoveGroup
    move_group.set_pose_target(target_pose)

    # Plan and execute the motion
    plan = move_group.go(wait=True)
    
    current_pose = move_group.get_current_pose().pose
    rospy.loginfo(f"Target Pose: {current_pose}")

    # Ensure that there is no residual movement
    move_group.stop()

    # Clear the target pose
    move_group.clear_pose_targets()

    rospy.loginfo("Motion complete!")

    # Shut down MoveIt cleanly
    moveit_commander.roscpp_shutdown()

if __name__ == "__main__":
    main()
