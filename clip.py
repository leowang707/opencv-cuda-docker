#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import CompressedImage

def callback(data):
    with open("output.h264", "ab") as f:
        f.write(data.data)

def listener():
    rospy.init_node('h264_listener', anonymous=True)
    rospy.Subscriber("/camera_stitched/color/image_raw/avc", CompressedImage, callback)
    rospy.spin()

if __name__ == '__main__':
    listener()
