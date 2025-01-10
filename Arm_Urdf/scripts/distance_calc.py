#!/usr/bin/env python3

import rospy
import tf
import math

def calculate_distance(link1, link2):
    rospy.init_node('cam_distance')
    listener = tf.TransformListener()

    try:
        # Wait for the transform to be available
        listener.waitForTransform(link1, link2, rospy.Time(0), rospy.Duration(5.0))
        
        # Lookup the transform between the two links
        (translation, rotation) = listener.lookupTransform(link1, link2, rospy.Time(0))
        
        # Extract translation values
        x, y, z = translation
        
        # Calculate Euclidean distance
        distance = math.sqrt(x**2 + y**2 + z**2)
        return distance
    except (tf.Exception, tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
        rospy.logerr(f"Transform not found between {link1} and {link2}: {e}")
        return None


if __name__ == '__main__':
    link1 = "camera_link1"
    link2 = "camera_link2"
    while not rospy.is_shutdown():    
        distance = calculate_distance(link1, link2)
        if distance is not None:
            rospy.loginfo(f"Distance between {link1} and {link2}: {distance:.2f} meters")



