import rclpy # Python ros2 interface

from geometry_msgs.msg import Twist, Vector3 # Twist message type import
from sensor_msgs.msg import Image
from rclpy.qos import qos_profile_sensor_data

from cv_bridge import CvBridge # Package to convert between ROS and OpenCV Images
import cv2 # OpenCV library
import numpy as np

import time

test_list = ['F', 'L', 'R', 'B'] # List of commands for testing
capture_shape = [720,1280] # Dimensions of the capture, it should have been width, then height, but I can't be bothered to go through an change it

forward_margin = 0.3 # This decides how close the robot should be to the object when driving forward
side_margin = 0.425 # This decides how much the object may deviate from the center, before the robot turns


# Class for publishing twist messages to the cmd_vel topic, based on the camera feed
# The twist messages are picked up by the differential controller and translated to the robot
class MRController:
    # Initiation parameters
    def __init__(self): 
        self.node = rclpy.create_node("MRCont") # Create a node called MRCont
        self.pub_twist = self.node.create_publisher(Twist, 'cmd_vel', 10) # Create the twist publisher

        self.sub_img = self.node.create_subscription(Image, '/camera/image_raw', self.image_callback, 10) # Create subscriber to the drone feed
        self.sub_img
        self.br = CvBridge() # Create bridge that converts ros image messages to opencv images
        self.current_image = np.zeros(capture_shape) # Initiate the image variable

        publish_period_sec = 0.2 # How often does the publisher publish              # self.get_param('publish_period', float, 0.2)
        self.tmr_twist = self.node.create_timer(publish_period_sec, self.on_tmr) # Creates timer that activates with a set interval

        self.move_dir = 'S' # The state of the robot, that decides how it is moving
        self.center_distance = 0 # Distance to the center of the detected object
        self.edge_distance = 0 # Distance to the edge of the detected object
        self.current_linear = [0, 0, 0] # Current linear velocity commands
        self.current_angular = [0, 0, 0] # Current angular velocity commands
        # self.i = 0

    # This function runs every time a new image is published to /camera/image_raw
    def image_callback(self, data):
        start_time = time.time() # Begins the timer
        self.current_image = self.br.imgmsg_to_cv2(data) # Convert from ros2 image msg to opencv image

        self.img_bgr = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB) # Converts the image to bgr format
        img_hsv = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2HSV) # Converts image to hsv

        h,s,c = cv2.split(img_hsv) # Split image and store hue colour channel

        mask = cv2.inRange(h, 1, 20) # Isolate box2 coloured objects
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) # Convert the objects mask to bgr for cosmetic purposes

        M = cv2.moments(mask) # use the moment function to analyse the blobs in the image
        
        # If an object is detected, then:
        if M["m00"] != 0:
            # Calculate x,y coordinate of center
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            # find the distance from the robot (center of the image), to the  center of the object
            self.center_distance = ((cX - capture_shape[1]/2) ** 2 + (cY - capture_shape[0]/2) ** 2) ** 0.5

            # Calculate the slope of the line from the robot to the center
            a1 = cX - capture_shape[1]/2
            a2 = cY - capture_shape[0]/2

            # Avoid division by zero
            if a1 == 0:
                a1= 0.01
            
            # Increase edge search detail if the object is directly in front or behind
            if abs(a1) < 10:
                detail = 0.05
            else:
                detail = 1

            # Normalise the slope
            a2 = a2/a1
            a1 = 1

            # Invert slope when on the right side of the robot
            if cX>capture_shape[1]/2:
                a1 = -a1 
                a2 = -a2

            edge = [-10,-10] # Variable to hold edge location
            n = 0 # Iterating variable
            check_loc = [int(capture_shape[1]/2 - a1*n), int(capture_shape[0]/2 - a2*n)] # Location for next edge check

            # iterate along the center-robot line until the object edge is found, or image border
            while check_loc[0] > 0 and check_loc[0] < capture_shape[1]-1 and check_loc[1] > 0 and check_loc[1] < capture_shape[0]-1:
                # If there is an object at check_loc, then save the location and break
                if mask_bgr[(check_loc[1],check_loc[0],0)].any() != 0:
                    edge = check_loc
                    break
                # Iterate n and calculate new check location
                n+=detail
                check_loc = [int(capture_shape[1]/2 - a1*n), int(capture_shape[0]/2 - a2*n)]
            
            # Calculate the distance from robot to edge of object
            self.edge_distance = ((edge[0] - capture_shape[1]/2) ** 2 + (edge[1] - capture_shape[0]/2) ** 2) ** 0.5

            # Draw the turn direction grid
            cv2.line(mask_bgr, (int(capture_shape[1]*side_margin), 0), (int(capture_shape[1]*side_margin), capture_shape[0]), (255,0,0), 2)
            cv2.line(mask_bgr, (int(capture_shape[1]*(1-side_margin)), 0), (int(capture_shape[1]*(1-side_margin)), capture_shape[0]), (255,0,0), 2)
            # cv2.line(mask_bgr, (int(capture_shape[1]*side_margin), int(capture_shape[0]*forward_margin)), (int(capture_shape[1]*(1-side_margin)), int(capture_shape[0]*forward_margin)), (255,0,0), 2)
            cv2.line(mask_bgr, (int(capture_shape[1]*side_margin), int(capture_shape[0]*0.5)), (int(capture_shape[1]*(1-side_margin)), int(capture_shape[0]*0.5)), (255,0,0), 2)
            
            # Draw robot-center line, center dot and edge dot
            cv2.line(mask_bgr, (cX, cY), (int(capture_shape[1]/2), int(capture_shape[0]/2)), (0,0,255), 1)
            cv2.circle(mask_bgr, (cX, cY), 5, (0,0,255), -1)
            cv2.circle(mask_bgr, edge, 5, (0,255,0), -1)

            # Set a movement direction state based on the loaction of the center
            if cX < capture_shape[1]*side_margin:   #Turn left if the object is left of center
                self.move_dir = 'L'
            elif cX > capture_shape[1]*(1-side_margin): #Turn right if the object is right of center
                self.move_dir = 'R'
            elif cY > capture_shape[0]*0.5: #Turn right if the object is behind the robot
                self.move_dir = 'R'
            elif self.edge_distance > 100: #Drive forwards towards the object if it is not too close
                self.move_dir = 'F'
            else:
                self.move_dir = 'S'

            endtime = round((time.time() - start_time)*1000,2) # Stop the timer

            # Write info on the screen
            cv2.putText(mask_bgr, self.move_dir, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
            cv2.putText(mask_bgr, str(int(self.center_distance)), (20,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
            cv2.putText(mask_bgr, str(int(self.edge_distance)), (20,120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
            cv2.putText(mask_bgr, str(endtime)+" ms", (20,160), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)


        cv2.imshow("camera", mask_bgr)
        cv2.waitKey(1)

    
    def get_param(self, name, expected_type, default):
        print("get_params")
        param = self.node.get_parameter(name)
        value = param.value
        if isinstance(value, expected_type):
            print("get_params - if")
            return value
        else:
            print("get_params - else")
            self.logger.warn(
                'Parameter {}={} is not a {}. Assuming {}.'.format(
                    param.name,
                    param.value,
                    expected_type,
                    default),
            )
            return default
        
    def on_tmr(self):
        # print("on_tmr")
        
        self.current_linear = [0, 0, 0]
        self.current_angular = [0, 0, 0]


        print("Moving: " + self.move_dir)
        if self.move_dir == 'F':
            self.current_linear[0] += 0.5
        elif self.move_dir == 'B':
            self.current_linear[0] -= 0.5
        elif self.move_dir == 'L':
            self.current_angular[2] += 1
            self.current_linear[0] += 0.1
        elif self.move_dir == 'R':
            self.current_angular[2] -= 1
            self.current_linear[0] += 0.1


        twist = Twist(
            linear=Vector3(
                x=float(self.current_linear[0]),
                y=float(self.current_linear[1]),
                z=float(self.current_linear[2]),
            ),
            angular=Vector3(
                x=float(self.current_angular[0]),
                y=float(self.current_angular[1]),
                z=float(self.current_angular[2]),
            )
        )
        self.pub_twist.publish(twist)

    # Spin the node
    def spin(self):
        print("spin")
        while rclpy.ok():
            rclpy.spin(self.node) 
            # Basically means run the node until it is stopped
    

def main(args=None):
    rclpy.init(args=args)
    node = MRController()
    node.spin()


if __name__ == '__main__':
    main()