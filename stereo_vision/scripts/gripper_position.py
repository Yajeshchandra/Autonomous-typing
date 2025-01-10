#!/usr/bin/env python3

import rospy
import tf

def get_gripper_position(reference_frame, gripper_link):
    listener = tf.TransformListener()

    try:
        # Wait for the transform to be available
        listener.waitForTransform(reference_frame, gripper_link, rospy.Time(0), rospy.Duration(5.0))
        
        # Lookup the transform between the two frames
        (translation, rotation) = listener.lookupTransform(reference_frame, gripper_link, rospy.Time(0))
        
        # Extract translation values
        x, y, z = translation
        
        return x, y, z
    except (tf.Exception, tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
        rospy.logerr(f"Transform not found between {reference_frame} and {gripper_link}: {e}")
        return None



if __name__ == '__main__':
    reference_frame = "world"
    gripper_link = "camera_link2"
    rospy.init_node('gripper_position')
    
    rate = rospy.Rate(2)  
    
    while not rospy.is_shutdown():    
        position = get_gripper_position(reference_frame, gripper_link)
        if position:
            x, y, z = position
            rospy.loginfo(f"Position of {gripper_link} relative to {reference_frame}: x={x:.2f}, y={y:.2f}, z={z:.2f}")
        else:
            rospy.logwarn("Could not get gripper position")
        rate.sleep()  # Sleep to maintain the desired rate