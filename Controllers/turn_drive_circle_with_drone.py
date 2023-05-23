import rclpy # Python ros2 interface

from std_msgs.msg import String
from geometry_msgs.msg import Twist, Vector3 # Twist message type import
from sensor_msgs.msg import Image
from rclpy.qos import qos_profile_sensor_data

from cv_bridge import CvBridge # Package to convert between ROS and OpenCV Images
import cv2 # OpenCV library
import numpy as np
from simple_pid import PID

import math
import time
import copy

test_list = ['F', 'L', 'R', 'B'] # List of commands for testing
capture_shape = [720,960] # Dimensions of the capture, it should have been width, then height, but I can't be bothered to go through an change it

forward_margin = 0.3 # This decides how close the robot should be to the object when driving forward
side_margin = 0.425 # This decides how much the object may deviate from the center, before the robot turns
circle_time = 1000000

search_vel_pid = [0.005,0,0] # PID parameters for the linear movement of the search function
search_vel_lim = [-0.1,0.1] # Limits for the PID controller for the linear movement of the search function

search_ang_pid = [0.03,0,0] # PID parameters for the angular movement of the search function
search_ang_lim = [-1,1] # Limits for the PID controller for the angular movement of the search function

circle_ang_pid = [0.02,0,0] # PID parameters for the angular movement of the follow function
circle_ang_lim = [-1,1] # Limits for the PID controller for the angular movement of the follow function

# Hue threshold for various targets
thresh_trash = [72, 75] #75,80
thresh_bag = [85, 98]
thresh_sprite = [54,70]
thresh_poster = [150,180]
thresh_red = [170,180]
thresh_purple = [120,160]
thresh_green = [70,78]
thresh = [thresh_trash, thresh_bag, thresh_sprite, thresh_poster, thresh_green, thresh_purple, thresh_red]

# Class for publishing twist messages to the cmd_vel and cmd_twist topics, based on the camera feed
# The cmd_vel twist messages are picked up by the differential controller and translated to the robot
# The cmd_twist twist messages are used by the connector node to control the drone's movement
class MRController:
    # Initiation parameters
    def __init__(self): 
        self.node = rclpy.create_node("MRCont") # Create a node called MRCont
        self.pub_twist = self.node.create_publisher(Twist, 'cmd_vel', 10) # Create the twist publisher
        self.pub_drone = self.node.create_publisher(Twist, 'drone_vel', 10) # Create the twist publisher

        self.sub_img = self.node.create_subscription(Image, '/camera/image_raw', self.image_callback, 10) # Create subscriber to the drone feed
        self.sub_img
        self.br = CvBridge() # Create bridge that converts ros image messages to opencv images
        self.current_image = np.zeros(capture_shape) # Initiate the image variable

        self.sub_img = self.node.create_subscription(String, 'RC', self.RC_callback, 10)

        publish_period_sec = 0.1 # How often does the publisher publish 
        self.tmr_twist = self.node.create_timer(publish_period_sec, self.on_tmr) # Creates timer that activates with a set interval

        self.move_dir = 'S'                 # The state of the robot, that decides how it is moving
        self.current_linear = [0, 0, 0]     # Current linear velocity commands
        self.current_angular = [0, 0, 0]    # Current angular velocity commands
        self.state = "search"               # Variable that controls the state of the robot (search, circle)
        self.circle_counter = 0             # Variable is used to time the circling funciton
        self.circle_turn90 = True           # Used to turn 90 degrees when switching between circle and search
        self.search_state = "turn"          # Variable that controls the turn and drive functionality (turn, drive)

        self.center_distance = 0            # Distance to the center of the detected object
        self.edge_distance = 0              # Distance to the edge of the detected object

        self.robot_loc = [capture_shape[1]/2,capture_shape[0]/2]    # Robot location
        self.robot_forw = [0,-10]           # Robot forward vector
        self.object_loc = [capture_shape[1]/2,0]    # Object location
        self.drone_diff = [0,0,0]           # Drone movement commands
        self.current_distance = 0           # Distance to the fire
        self.dist_conv = 1                  # Pixel to cm conversion

        self.detect_counter = 0             # Counter for successful AR-tag detections
        self.total_counter = 0              # Counter for total AR-tag detection attempts

        self.manual = False                 # Changes between automatic and manual driving

        self.data = open("data.txt", "w")   # Create file for logging data
        all_data = "Total, Detected, Lin. vel., Ang.vel., Roll, Pitch, Fire dist., Drone offset x, Drone offset y, Conversion, Process time\n" # Headers for the columns of data
        self.data.write(all_data)           # Write column headers to the file

        self.process_time = 0               # Variable used to record process time for data logging

        # P controller for the linear velocity, based on the distance to the object
        self.pidVel = PID(search_vel_pid[0],search_vel_pid[1],search_vel_pid[2],setpoint = 110)
        self.pidVel.output_limits = (search_vel_lim[0],search_vel_lim[1])

        # P controller for the angular velocity, based on the angle between the robot and the object
        self.pidAng = PID(search_ang_pid[0],search_ang_pid[1],search_ang_pid[2], setpoint = 0)
        self.pidAng.output_limits = (search_ang_lim[0],search_ang_lim[1])

        # P controller with a target value that is the same as the distance for the approach
        self.followPID = PID(circle_ang_pid[0],circle_ang_pid[1],circle_ang_pid[2], setpoint=0)
        self.followPID.output_limits = (circle_ang_lim[0],circle_ang_lim[1])

        # PID controller for the drone's roll, based on the robot's location on the x axis in the image
        self.rollPID = PID(-1,-0.001,0.001, setpoint=0) 
        self.rollPID.output_limits = (-1,1)

        # # PID controller for the drone's pitch, based on the robot's location on the y axis in the image
        self.pitchPID = PID(-1,-0.001,0.001, setpoint=0)
        self.pitchPID.output_limits = (-1,1)

    # This function runs every time a new image is published to /camera/image_raw
    def image_callback(self, data):
        start_time = time.time() # Begins the timer

        self.current_image = self.br.imgmsg_to_cv2(data) # Convert from ros2 image msg to opencv image

        self.img_bgr = self.current_image 
        
        # Reset the linear and angular commands before calculating new ones
        self.current_linear = [0, 0, 0]
        self.current_angular = [0, 0, 0]

        try:
            # Run the AR-tag detection function
            self.robot_loc, self.robot_forw = self.ar_detection(self.img_bgr) 

            # Calculate drone movement
            self.drone_diff = self.droneMovement()
            
            # State machine if-chain. Switches between "search" and "circle"
            if self.state == "search": # "search" represents searching for and moving to the fire
                # Run fire detection and P controller for searching
                movement, self.object_loc = self.basicFireDetection(self.img_bgr) 

                # Draws the line from the robot to the object, and a dot representing the closest pixel
                cv2.line(self.img_bgr, [int(self.object_loc[0]), int(self.object_loc[1])], [int(self.robot_loc[0]), int(self.robot_loc[1])], (0,0,255), 1)
                cv2.circle(self.img_bgr, [int(self.object_loc[0]), int(self.object_loc[1])], 5, (0,0,255), -1)
                
                cv2.line(self.img_bgr, [20,20], [20, int(20+100*1.4)], (0,0,0), 10) # Draws the line for the circle counter
            elif self.state == "circle": # "circle" represents circling around the fire
                # Turn on the spot for the first 7 iterations to avoid driving into the fire
                if self.circle_counter > 7:
                    self.circle_turn90 = False
                
                # Run fire detection and P controller for circling
                movement, self.object_loc = self.followFire(self.img_bgr, self.circle_turn90) 

                # Iterate the circle_counter
                self.circle_counter = self.circle_counter +1 

                # Draws the line from the robot to the object
                cv2.line(self.img_bgr, [int(self.object_loc[0]), int(self.object_loc[1])], [int(self.robot_loc[0]), int(self.robot_loc[1])], (255,0,0), 1)
                cv2.circle(self.img_bgr, [int(self.object_loc[0]), int(self.object_loc[1])], 5, (255,0,0), -1)

                # Draws the progress bar for the circle counter
                cv2.line(self.img_bgr, [20,20], [20, int(20+100*1.4)], (0,0,0), 10)
                cv2.line(self.img_bgr, [20,20], [20, int(20+((self.circle_counter/circle_time)*100)*1.4)], (255,255,255), 10)

                if self.circle_counter > circle_time: 
                    # If the robot has circled for "circle_time" iterations, switch to "search" to face the fire and reset
                    self.state = "search"
                    self.circle_counter = 0
                    self.circle_turn90 = True

            # Extract linear and angular movement commands and save them
            if movement:
                self.current_linear = movement[0]
                self.current_angular = movement[1]
        except:
            pass    

        endtime = round((time.time() - start_time)*1000,2) # Stop the timer
        self.process_time = endtime

        # Write info on the screen
        cv2.putText(self.img_bgr, self.state, (40,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(self.img_bgr, str(endtime)+" ms", (40,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(self.img_bgr, "speed: " + str(round(self.current_linear[0],2)), (40,120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(self.img_bgr, "turn: " + str(round(self.current_angular[2],2)), (40,160), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

        # Draw the forward line on the robot
        cv2.line(self.img_bgr, [int(self.robot_loc[0]), int(self.robot_loc[1])], [int(self.robot_loc[0]+self.robot_forw[0]*2), int(self.robot_loc[1]+self.robot_forw[1]*2)], (255,0,0), 1)
        cv2.circle(self.img_bgr, [int(960/2),int(720/2)], 5, (0,0,255), -1)

        # Show image
        cv2.imshow("camera", self.img_bgr)
        cv2.waitKey(1)
    
    ##### AR tag detection functions BEGIN #####
    # Main function for generating AR-tags
    def ar_detection(self, camera_feed):
        # Generate contours to detect corners of the tag
        final_contour_list = self.contour_generator(camera_feed) 

        # Count the attempt at detecting AR-tags
        self.total_counter = self.total_counter + 1

        # If there are no conours, end and return 0's
        if len(final_contour_list) == 0:
            return [0,0], [0,0]
        
        # Creates an empty countour square description
        dim = 200
        p1 = np.array([
            [0, 0],
            [dim - 1, 0],
            [dim - 1, dim - 1],
            [0, dim - 1]], dtype="float32")

        # Go through all found contours
        for i in range(len(final_contour_list)):
            # Draw current contours
            cv2.drawContours(self.img_bgr, [final_contour_list[i]], -1, (255, 0, 0), 2)

            # Extractg the corner coordinates of the contour
            c_rez = final_contour_list[i][:, 0]

            # Create homography matrix to convert from contour square to corrected perspective
            H_matrix = self.homograph(p1, self.order(c_rez))

            # Perspective correct the ar tag using the homography matrix
            tag = cv2.warpPerspective(camera_feed, H_matrix, (200, 200))

            # Create a b/w copy of the tag
            tag1 = cv2.cvtColor(tag, cv2.COLOR_BGR2GRAY)

            # Get tag id and location of orientation square
            decoded, location = self.id_decode(tag1)
            
            # Draw points of detection for debugging
            cv2.circle(tag, [62,62], 1, (255,0,0), 2)
            cv2.circle(tag, [62,137], 1, (255,0,0), 2)
            cv2.circle(tag, [137,62], 1, (255,0,0), 2)
            cv2.circle(tag, [137,137], 1, (255,0,0), 2)

            cv2.circle(tag, [87,87], 1, (0,0,255), 2)
            cv2.circle(tag, [87,112], 1, (0,0,255), 2)
            cv2.circle(tag, [112,87], 1, (0,0,255), 2)
            cv2.circle(tag, [112,112], 1, (0,0,255), 2)

            # If a location has been gathered
            if not location == None:
                # Calculate AR-tag center, forward vector, and cm/pixel conversion
                cent, forw, self.dist_conv = self.tag_location(location, self.order(c_rez))
                # Count successful detection
                self.detect_counter = self.detect_counter + 1
                return cent, forw

            else:
                print("Location: NONE")

    # Generate and sort contours in the image, returns the outer square of the ar tag
    def contour_generator(self, frame):
        # Create b/w of the video feed
        test_img1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Blur image for cleaner edges
        test_blur = cv2.GaussianBlur(test_img1, (3, 3), 0)
        # Edge detection
        edge = cv2.Canny(test_blur, 75, 200)
        # Make copy of edge image
        edge1 = copy.copy(edge)

        # Display the edge detection image
        cv2.imshow("Edge", edge)

        # Create empty list
        contour_list = list()

        # Find contours of the edge image
        cnts, h = cv2.findContours(edge1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # |   \___ Hieracy
        #  \______ Contours
        
        # This part crashes if there are no contours, so a try/execept was created
        try:
            # Create empty list of indexes
            index = list()
            # Loop through all hieracies and save the ones with a hieracy higher than -1 (those whith parents)
            for hier in h[0]:
                if hier[3] != -1:
                    index.append(hier[3])

            # loop over the contours that pass the hieracy test
            for c in index:
                # Calculate the length of the edges of a contour
                peri = cv2.arcLength(cnts[c], True)
                # Approximate the number of edges of the contour given a polygonal shape
                approx = cv2.approxPolyDP(cnts[c], 0.02 * peri, True)

                # If a contour has more than 4 edges (looking for the center part of the AR-tag)
                if len(approx) > 4:
                    # Calculate the length of the edges of a contour
                    peri1 = cv2.arcLength(cnts[c - 1], True)
                    # Approximate the number of edges of the contour given a polygonal shape
                    corners = cv2.approxPolyDP(cnts[c - 1], 0.02 * peri1, True)
                    # Add the contour to the list
                    contour_list.append(corners)

            # Create NEW empty list
            new_contour_list = list()
            # Only keep those contours with length four (black outer square of the AR-tag)
            for contour in contour_list:
                if len(contour) == 4:
                    new_contour_list.append(contour)
            
            # Create *FINAL* empty list
            final_contour_list = list()
            # Only keep those contours with an area of less than 8000 and more than 10
            for element in new_contour_list:
                if cv2.contourArea(element) < 8000 and cv2.contourArea(element) > 10:
                    final_contour_list.append(element)

            return final_contour_list
        except:
            return 0

    # Order the corners of the square in counterclockwise order
    def order(self, pts):
        # Create empty four sided contour
        rect = np.zeros((4, 2), dtype="float32")

        # Add the coordinates and set min and max as corners 0 and 2
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        # Subtract the coordinates to and set min and max to corners 1 and 3
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        # return the ordered coordinates
        return rect

    # Create a homography matrix that converts from one frame to another
    def homograph(self, p, p1):
        A = []
        p2 = self.order(p)              # Order the p contour

        # Go through all elements of p1
        for i in range(0, len(p1)):
            x, y = p1[i][0], p1[i][1]
            u, v = p2[i][0], p2[i][1]
            A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
            A.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
        A = np.array(A)
        U, S, Vh = np.linalg.svd(A)     # Singular value decomposition
        l = Vh[-1, :] / Vh[-1, -1]
        h = np.reshape(l, (3, 3))
        # print(l)
        # print(h)
        return h

    # Decode the ID and orientation of the AR-tag
    def id_decode(self, image):
        # create binary from the input image
        ret, img_bw = cv2.threshold(image, 170, 255, cv2.THRESH_BINARY)

        # Create a copy of the AR-tag and draw detection points on it for debugging
        bw_copy = img_bw
        bw_copy = cv2.cvtColor(bw_copy, cv2.COLOR_GRAY2BGR)
        cv2.circle(bw_copy, [62,62], 1, (255,0,0), 2)
        cv2.circle(bw_copy, [62,137], 1, (255,0,0), 2)
        cv2.circle(bw_copy, [137,62], 1, (255,0,0), 2)
        cv2.circle(bw_copy, [137,137], 1, (255,0,0), 2)

        cv2.circle(bw_copy, [87,87], 1, (0,0,255), 2)
        cv2.circle(bw_copy, [87,112], 1, (0,0,255), 2)
        cv2.circle(bw_copy, [112,87], 1, (0,0,255), 2)
        cv2.circle(bw_copy, [112,112], 1, (0,0,255), 2)
        cv2.imshow("threshold", bw_copy)

        # pixel value of the corner pixel that we are searching for
        corner_pixel = 255

        # Crop to isolate the center square
        cropped_img = img_bw[50:150, 50:150]

        # Select center pixels of the four id blocks
        block_1 = cropped_img[37, 37]
        block_3 = cropped_img[62, 37]
        block_2 = cropped_img[37, 62]
        block_4 = cropped_img[62, 62]

        # In case you wondered what value white is:
        white = 255

        # Setting the values for each ID square: white = 1, black = 0
        if block_3 == white:
            block_3 = 1
        else:
            block_3 = 0
        if block_4 == white:
            block_4 = 1
        else:
            block_4 = 0
        if block_2 == white:
            block_2 = 1
        else:
            block_2 = 0
        if block_1 == white:
            block_1 = 1
        else:
            block_1 = 0

        # Return the tag's id, corrected based on orientation (where the white corner is)
        if cropped_img[89, 89] == corner_pixel:
            return list([block_3, block_4, block_2, block_1]), "BL"
        elif cropped_img[10, 89] == corner_pixel:
            return list([block_4, block_2, block_1, block_3]), "BR"
        elif cropped_img[12, 10] == corner_pixel:
            return list([block_2, block_1, block_3, block_4]), "TR"
        elif cropped_img[89, 10] == corner_pixel:
            return list([block_1, block_3, block_4, block_2]), "TL"

        # Otherwise return empty
        return None, None

    # Calculate the tag's location an orientation in the image
    def tag_location(self, orient, corners):
        front_loc = [0,0]   # creates empty fron facing point

        # Calculate the forward facing point as the middle of two corners, based on the orientation
        if orient == "TR":
            front_loc = [(corners[0][0]+corners[3][0])/2, (corners[0][1]+corners[3][1])/2]
        elif orient == "TL":
            front_loc = [(corners[2][0]+corners[3][0])/2, (corners[2][1]+corners[3][1])/2]
        elif orient == "BL":
            front_loc = [(corners[1][0]+corners[2][0])/2, (corners[1][1]+corners[2][1])/2]
        elif orient == "BR":
            front_loc = [(corners[0][0]+corners[1][0])/2, (corners[0][1]+corners[1][1])/2]
        
        # Set the center coordinates as the average position of the four corners
        center_loc = [(corners[0][0]+corners[1][0]+corners[2][0]+corners[3][0])/4, (corners[0][1]+corners[1][1]+corners[2][1]+corners[3][1])/4]
        
        # Set the forward vector as going from the center to the forward point
        forward_vector = [front_loc[0]-center_loc[0], front_loc[1]-center_loc[1]]

        # Calculate the sidelength of the square
        side_length = math.sqrt(pow(corners[0][0]-corners[1][0],2) + pow(corners[0][1]-corners[1][1],2))
        # Calculate conversion between the side length in pixels and in cm 
        convertion = 15/side_length

        return center_loc, forward_vector, convertion
    #####  AR tag detection functions END  #####

    # Calculate angle between two vectors
    def find_angle(self, sVector, eVector):
        # Components of the vectors
        a = sVector[0]
        b = sVector[1]
        c = eVector[0]
        d = eVector[1]

        dotProduct = a*c + b*d # Take the dot product of the vectors
        
        modOfVector1 = math.sqrt( a*a + b*b)*math.sqrt(c*c + d*d) # Pythagoras theorem

        # Handles the zero case
        if modOfVector1 == 0:
            modOfVector1 = 0.01

        angle = math.atan2( a*d - b*c, a*c + b*d ) # Calculate the angle using atan2 to allow negative angle

        angleInDegree = math.degrees(angle) # Convert to degrees

        return angleInDegree

    # Calculate the tangent angle to the fire, based on the distance to the fire
    def tangent_angle(self, object, dist, target):
        rob_obj = object - self.robot_loc   # Calculate the vector from the robot to the fire
        angle = -self.find_angle(self.robot_forw, rob_obj) +90  # Calculate the angle to the fire

        err = target - dist                 # Calculate the error between the target distanceand current distance

        heading = -(err + angle)            # Calculate the heading as the sum of the error and angle

        return heading
        
    # Fire detection and proportional control approach
    def basicFireDetection(self, camera_feed):
        img_blur = cv2.GaussianBlur(camera_feed, (3, 3), 0) # Blur the image slightly to reduce noise
        image = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HLS)   # Convert image to HLS
        
        kernel = np.ones((19, 19), np.uint8) # Used in noise reduction
        
        threshold = 110       # How far the robot should be from the fire
        nearest_point = [0,0] # Variable for storing the neares object pixel

        (h, l, s) = cv2.split(image) # Split the channels of the image and save the hue channel
        image = cv2.inRange(h, thresh[5][0], thresh[5][1]) # 85,98 # threshold the image to find the fire
        image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel) # Noise reduction

        cv2.imshow("something",image) # Show fire detection image

        shortest_dist = 1000000 # arbitrarily high starting value for shortest distance to object
        nonzero = cv2.findNonZero(image) # Find locations of all non-zero pixels in image

        nearest_point = [0,0]
        # If there is no object it gets angry, so I did a try/except to handle it
        try:
            # Measure distances to all nonzero entries
            distances = np.sqrt((nonzero[:,:,0] - self.robot_loc[0]) ** 2 + (nonzero[:,:,1] - self.robot_loc[1]) ** 2) 
            nearest_index = np.argmin(distances) # Find the index of the closest pixel
            nearest_point = nonzero[nearest_index] # Get coordinated of nearest pixel
            # Calculate shortest distance... again
            shortest_dist = np.sqrt((nearest_point[0][0] - self.robot_loc[0]) ** 2 + (nearest_point[0][1] - self.robot_loc[1]) ** 2) 
        except:
            # If no object is present, set nearest pixel to the center pixel
            nearest_point = [[image.shape[0] / 2, image.shape[1] / 2]]
            shortest_dist = 1000000
        
        self.current_distance = shortest_dist # Save to variable used for data logging

        # Reset turn and speed before calculating new values
        turn = 0
        targetSpeed = 0
        
        # Calculate the vectors for robot to object, and robot forward
        rob_obj = [nearest_point[0][0]-self.robot_loc[0], nearest_point[0][1]-self.robot_loc[1]]

        object_angle = self.find_angle(rob_obj, self.robot_forw) # Calculate angle to the object

        # If the angle to the object is more than 20 degrees, begin turning
        if abs(object_angle) > 20:
            self.search_state = "turn"

        # If the remaining angel is more than three degees and turn mode is active, turn until the angle is less, then change to drive mode
        if self.search_state == "turn" and abs(object_angle) > 3:# and not(abs(object_angle) == 0): # abs(object_angle) != 0 and abs(object_angle) > 3 and 
            turn = self.pidAng(-object_angle) # Set turn value based on the angle to the object
        else:
            # When the robot has turned, it switches to drive
            print("Changing to drive")
            self.search_state = "drive"
        
        # Drive forward while in drive mode and the distance to object is less than the threshold, then change to circling mode
        if self.search_state == "drive":
            if shortest_dist > threshold:
                # If the robot is further away than the threshold
                targetSpeed = -self.pidVel(shortest_dist) # Set speed value based on the distance to the nearest point
            else:
                # When the robot has reached its destination
                print("Search done!")
                self.search_state = "turn"  # Change the drive state back to "turn"
                self.state = "circle"       # Change the state of the robot to "circle"

        return [[targetSpeed,0,0], [0,0,turn]], nearest_point[0] # Return the twist msg components 

    # Circle around fire using P-controller 
    def followFire(self, camera_feed, turn_state):

        kernel = np.ones((19,19), np.uint8) # Kernel used in noise reduciton
        
        # If turn state is true then turn left. Used when switching to circle
        if turn_state:
            return [[0,0,0], [0,0,2]], self.robot_loc
        
        img_blur = cv2.GaussianBlur(camera_feed, (3, 3), 0) # Blur the image slightly to reduce noise
        image = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HLS)   # Convert image to HLS
        (h, l, s) = cv2.split(image)                        # Split channels and save hue channel 
        image = cv2.inRange(h, thresh[5][0], thresh[5][1])  # Threshhold image for objects
        image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel) # Noise reduction (Remove this later perhaps)
        cv2.imshow("something",image)

        center = self.robot_loc
        
        nonzero = cv2.findNonZero(image) # Find all non-zero pixels in the image
        
        try:
            # Measure distances to all nonzero entries
            distances = np.sqrt((nonzero[:,:,0] - center[0]) ** 2 + (nonzero[:,:,1] - center[1]) ** 2) 
            nearest_index = np.argmin(distances) # Find the index of the closest pixel
            nearest_point = nonzero[nearest_index] # Get coordinated of nearest pixel
            # Calculate shortest distance... again
            shortest_dist = np.sqrt((nearest_point[0][0] - center[0]) ** 2 + (nearest_point[0][1] - center[1]) ** 2) 
        except:
            # If no object is present, set nearest pixel to the center pixel and distance to be 110
            nearest_point = [[image.shape[0] / 2, image.shape[1] / 2]]
            shortest_dist = 110
        
        # If no AR-tag has been detected, set the shortest distance to 0
        if self.robot_loc != [0,0]:
            self.current_distance = shortest_dist
        else:
            self.current_distance = 0

        target_angle = self.tangent_angle(nearest_point[0], shortest_dist, 110) # Calculate desired tangent angle value based on distance to object

        turn_pid = self.followPID(target_angle) # Calculate turning value based on the angle to the desired tangent angle

        return [[0.1,0,0], [0,0,turn_pid]], nearest_point[0] # Return twist msg components
        
    # Calculate drone offset and send to drone controller
    def droneMovement(self):
        # If no AR-tag has been detected, don't move 
        if self.robot_loc == [0,0]:
            return [0,0,0]
        
        # Calculate a value from -1 to 1 for x and y, based on the robot's position in the image, with [0,0] being at the center
        diffx = (self.robot_loc[0]-capture_shape[1]/2)/(capture_shape[1]/2)
        diffy = (self.robot_loc[1]-capture_shape[0]/2)/(capture_shape[0]/2)
        # The z (yaw) value is set to 0, as it causes too much disturbance
        diffz = 0 #self.find_angle([0,-1],self.robot_forw)/180

        PIDx = self.rollPID(diffx)  # Feed the x component through the roll PID controller
        PIDy = self.pitchPID(diffy) # Feed the y component through the pitch PID controller

        return [PIDx, -PIDy, diffz]

    # Callback function used when switching to and from remote manual control
    def RC_callback(self, data):

        cmd = data.data # Save the command

        # If RCON was recieved turn on remote control mode
        if cmd == "RCON":
            # A twist message of 0's is created
            vel_twist = Twist(
                linear=Vector3(
                    x=float(0),
                    y=float(0),
                    z=float(0),
                ),
                angular=Vector3(
                    x=float(0),
                    y=float(0),
                    z=float(0),
                )
            )
            # Publish the 0's twist messages to stop the robot moving
            self.pub_twist.publish(vel_twist)
            self.manual = True # Set the manual state to True 
        # If RCOFF was recieved turn off remote control mode
        elif cmd == "RCOFF":
            self.state = "search"       # Return to search mode
            self.circle_counter = 0     # Reset the circle counter
            self.circle_turn90 = True   # Reset turn90 Bool
            self.manual = False         # Set manual mode to False

    # Runs every set interval and sends twist message
    def on_tmr(self):
        # Write the collected data to a textfile
        all_data = str(self.total_counter)+", "+str(self.detect_counter)+", "+str(self.current_linear[0])+", "+str(self.current_angular[2])+", "+str(self.drone_diff[0])+", "+str(self.drone_diff[1])+", "+str(self.current_distance)+", "+str(self.robot_loc[0]-960/2)+", "+str(self.robot_loc[1]-720/2)+", "+str(self.dist_conv)+",0"+str(self.process_time)+"\n"
        self.data.write(all_data)

        # Create twist message from the latest linear and angular velocities
        vel_twist = Twist(
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

        # Create twist message with drone RPY
        drone_twist = Twist(
            linear=Vector3(
                x=float(0),
                y=float(0),
                z=float(0),
            ),
            angular=Vector3(
                x=float(self.drone_diff[0]),
                y=float(self.drone_diff[1]),
                z=float(self.drone_diff[2]),
            )
        )
        # If the robot is not in manual control mode, publish the commands for the Turtlebot
        if self.manual == False:
            self.pub_twist.publish(vel_twist)
        
        # Publish the commands for the Tello drone
        self.pub_drone.publish(drone_twist)
        
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