import rclpy # Python ros2 interface

from geometry_msgs.msg import Twist, Vector3 # Twist message type import
from sensor_msgs.msg import Image
from rclpy.qos import qos_profile_sensor_data

from cv_bridge import CvBridge # Package to convert between ROS and OpenCV Images
import cv2 # OpenCV library
import numpy as np
from simple_pid import PID

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

        publish_period_sec = 0.2 # How often does the publisher publish 
        self.tmr_twist = self.node.create_timer(publish_period_sec, self.on_tmr) # Creates timer that activates with a set interval

        self.move_dir = 'S'                 # The state of the robot, that decides how it is moving
        self.center_distance = 0            # Distance to the center of the detected object
        self.edge_distance = 0              # Distance to the edge of the detected object
        self.current_linear = [0, 0, 0]     # Current linear velocity commands
        self.current_angular = [0, 0, 0]    # Current angular velocity commands
        self.state = "search"               # Variable that controls the state of the robot (Search, circle)
        self.circle_counter = 0             # Variable is used to time the circling funciton
        self.circle_turn90 = True           # Used to turn 90 degrees when switching between circle and search

    # This function runs every time a new image is published to /camera/image_raw
    def image_callback(self, data):
        start_time = time.time() # Begins the timer

        self.current_image = self.br.imgmsg_to_cv2(data) # Convert from ros2 image msg to opencv image

        self.img_bgr = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB) # Converts the image to bgr format
        
        # Reset the linear and angular commands
        self.current_linear = [0, 0, 0]
        self.current_angular = [0, 0, 0]

        # State machine if-chain
        if self.state == "search": # "search" represents searching for and moving to the fire
            print("Searching")
            movement = self.basicFireDetection(self.img_bgr) # Run fire detection and P control
        elif self.state == "circle": # "circle" represents circling around the fire
            if self.circle_counter > 100: 
                # If the robot has circled for 100 iterations, face the fire and reset
                print("Circle: end")
                self.state = "search"
                self.circle_counter = 0
                self.circle_turn90 = True
            elif self.circle_counter > 5:
                # Turn on the spot for the first 5 iterations
                print("Circle")
                self.circle_turn90 = False
                
            movement = self.followFire(self.img_bgr, self.circle_turn90) # Run fire detection and P control
            self.circle_counter += 1 
            print(self.circle_counter)


        # Extract linear and angular movement commands
        if movement:
            self.current_linear = movement[0]
            self.current_angular = movement[1]
        
        endtime = round((time.time() - start_time)*1000,2) # Stop the timer

        # Write info on the screen
        cv2.putText(self.img_bgr, str(endtime)+" ms", (20,160), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

        # Show image
        cv2.imshow("camera", self.img_bgr)
        cv2.waitKey(1)

    # Honestly not sure what this does, I'm not using it
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
        
    # Fire detection and proportional control approach
    def basicFireDetection(self, camera_feed):
        # Setup
        image = cv2.cvtColor(camera_feed, cv2.COLOR_BGR2HLS) # Convert image to HLS
        running = True
        kernel = np.ones((5, 5), np.uint8) # Used in noise reduction
        threshold = 100 # Change this depending on what you need it as
        nearest_point = [0,0] # Variable for storing the neares object pixel
        center = np.array((image.shape[0] / 2, image.shape[1] / 2))  # Calculates the center of the image

        # P controller for the linear velocity, based on the distance to the object
        pidVel = PID(-0.01,0,0,setpoint = 100)
        pidVel.output_limits = (-0.5, 0.5)

        # P controller for the angular velocity, based on the distance from the object to the centerline
        pidAng = PID(0.01,0,0, setpoint = image.shape[1] / 2)
        pidAng.output_limits = (-1,1)

        # Main
        (h, l, s) = cv2.split(image)
        image = cv2.inRange(h, 10, 20)
        image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel) # Noise reduction (Remove this later perhaps)

        shortest_dist = 1000000 # arbitrarily high starting value for shortest distance to object
        nonzero = cv2.findNonZero(image) # Find locations of all non-zero pixels in image

        # If there is no object it gets angry, so I did a try/except to handle it
        try:
            distances = np.sqrt((nonzero[:,:,0] - center[1]) ** 2 + (nonzero[:,:,1] - center[0]) ** 2) # Measure distances to all nonzero entries
            nearest_index = np.argmin(distances) # Find the index of the closest pixel
            nearest_point = nonzero[nearest_index] # Get coordinated of nearest pixel
            shortest_dist = np.sqrt((nearest_point[0][0] - center[1]) ** 2 + (nearest_point[0][1] - center[0]) ** 2) # Calculate shortest distance... again
        except:
            # If no object is present, set nearest pixel to the center pixel
            nearest_point = [[image.shape[0] / 2, image.shape[1] / 2]]
            shortest_dist = 0
        
        # Reset turn and speed before calculating new values
        turn = 0
        targetSpeed = 0

        if abs(nearest_point[0][0] - image.shape[1]/2) > 0: # The if statement is redundant (it always runs), but i might want to reintroduce it so it stays
            turn = pidAng(nearest_point[0][0]) # Set turn value based on the x-coordinate of the nearest point
            print("turn  " + str(turn))
            
        if shortest_dist > threshold: # Check if the object is within the desired distance
            print("dist  " + str(shortest_dist))
            targetSpeed = pidVel(shortest_dist) # Set speed value based on the distance to the nearest point
            print("speed " + str(targetSpeed))
        elif abs(turn) < 0.05: # I the object is within acceptable distance and turn value is suitably small
            self.state = "circle" # Change the state of the robot to "circle"

        return [[targetSpeed,0,0], [0,0,turn]] # Return the twist msg components 

    # Circle around fire using P-controller 
    def followFire(self, camera_feed, turn_state):
        # Create PID cotroller with a target value that is the same as the distance for the approach
        followPID = PID(0.1,0,0, setpoint=100)
        followPID.output_limits = (-0.5,0.5)

        kernel = np.ones((5, 5), np.uint8) # Kernel used in noise reduciton

        # If turn state is true then turn left (five iterations gives approximately 90 degrees)
        if turn_state:
            return [[0,0,0], [0,0,1]]

        image = cv2.cvtColor(camera_feed, cv2.COLOR_BGR2HLS) # Convert image to HLS
        (h, l, s) = cv2.split(image) # Split channels and save hue channel 
        image = cv2.inRange(h, 10, 20) # Threshhold image for objects
        image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)         # Noise reduction (Remove this later perhaps)

        shortest_path = 1000000 # Arbitrarily high shortest path value
        center = np.array((image.shape[0] / 2, image.shape[1] / 2))  # Calculates the center of the image
        
        nonzero = cv2.findNonZero(image) # Find all non-zero pixels in the image
        distances = np.sqrt((nonzero[:,:,0] - center[1]) ** 2 + (nonzero[:,:,1] - center[0]) ** 2) # Calculate distance ot all non-zero pixels
        nearest_index = np.argmin(distances) # Find index of smallest distance
        nearest_point = nonzero[nearest_index] # Find location of closest pixel
        shortest_dist = np.sqrt((nearest_point[0][0] - center[1]) ** 2 + (nearest_point[0][1] - center[0]) ** 2) # Calculate shorest distance... again
 
        turn_pid = followPID(shortest_dist) # Calculate turn value based on distance to object

        return [[0.1,0,0], [0,0,turn_pid]] # Return twist msg components
        
    # Runs every set interval
    def on_tmr(self):
        # Create twist message from the latest linear and angular
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
        # Publish the twist message 
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