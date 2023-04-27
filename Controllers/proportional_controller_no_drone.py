import rclpy # Python ros2 interface

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

# ar_size = 36 #36 is good
# ar_tag = cv2.imread("Controllers/AR-tag/AR_tag_border.png")
# ar_tag = cv2.resize(ar_tag, (ar_size,ar_size), interpolation = cv2.INTER_AREA)

test_list = ['F', 'L', 'R', 'B'] # List of commands for testing
capture_shape = [480,640] # Dimensions of the capture, it should have been width, then height, but I can't be bothered to go through an change it

forward_margin = 0.3 # This decides how close the robot should be to the object when driving forward
side_margin = 0.425 # This decides how much the object may deviate from the center, before the robot turns

search_vel_pid = [0.001,0,0]
search_vel_lim = [-0.2,0.2]

search_ang_pid = [0.01,0,0]
search_ang_lim = [-1,1]

circle_ang_pid = [0.05,0,0]
circle_ang_lim = [-1,1]

# Class for publishing twist messages to the cmd_vel topic, based on the camera feed
# The twist messages are picked up by the differential controller and translated to the robot
class MRController:
    # Initiation parameters
    def __init__(self): 
        self.node = rclpy.create_node("MRCont") # Create a node called MRCont
        self.pub_twist = self.node.create_publisher(Twist, 'cmd_vel', 10) # Create the twist publisher

        # self.sub_img = self.node.create_subscription(Image, '/camera/image_raw', self.image_callback, 10) # Create subscriber to the drone feed
        # self.sub_img
        # self.br = CvBridge() # Create bridge that converts ros image messages to opencv images
        # self.current_image = np.zeros(capture_shape) # Initiate the image variable
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

        publish_period_sec = 0.1 # How often does the publisher publish 
        self.tmr_twist = self.node.create_timer(publish_period_sec, self.on_tmr) # Creates timer that activates with a set interval

        self.move_dir = 'S'                 # The state of the robot, that decides how it is moving
        self.current_linear = [0, 0, 0]     # Current linear velocity commands
        self.current_angular = [0, 0, 0]    # Current angular velocity commands
        self.state = "search"               # Variable that controls the state of the robot (Search, circle)
        self.circle_counter = 0             # Variable is used to time the circling funciton
        self.circle_turn90 = True           # Used to turn 90 degrees when switching between circle and search

        self.center_distance = 0            # Distance to the center of the detected object
        self.edge_distance = 0              # Distance to the edge of the detected object

        self.robot_loc = [capture_shape[1]/2,capture_shape[0]/2] # Initiating robot location
        self.robot_forw = [0,-10]
        self.object_loc = [capture_shape[1]/2,0] # Initiating object location

    # This function runs every time a new image is published to /camera/image_raw
    def image_callback(self, data):
        start_time = time.time() # Begins the timer

        # self.current_image = self.br.imgmsg_to_cv2(data) # Convert from ros2 image msg to opencv image

        self.img_bgr = data #cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB) # Converts the image to bgr format

        # self.img_bgr[int(capture_shape[0]/2-ar_size/2):int(capture_shape[0]/2+ar_size/2), int(capture_shape[1]/2-ar_size/2):int(capture_shape[1]/2+ar_size/2)] = ar_tag[:,:]
        
        # Reset the linear and angular commands
        self.current_linear = [0, 0, 0]
        self.current_angular = [0, 0, 0]
        try:
            self.robot_loc, self.robot_forw = self.ar_detection(self.img_bgr)
            # print(self.ar_detection(self.img_bgr))
            print(self.state)
            # State machine if-chain
            if self.state == "search": # "search" represents searching for and moving to the fire

                movement, self.object_loc = self.basicFireDetection(self.img_bgr) # Run fire detection and P control

                # Draws the line from the robot to the object
                cv2.line(self.img_bgr, [int(self.object_loc[0]), int(self.object_loc[1])], [int(self.robot_loc[0]), int(self.robot_loc[1])], (0,0,255), 1)
                cv2.circle(self.img_bgr, [int(self.object_loc[0]), int(self.object_loc[1])], 5, (0,0,255), -1)
                
                cv2.line(self.img_bgr, [20,20], [20, int(20+100*1.4)], (0,0,0), 10) # Draws the line for the circle counter
            elif self.state == "circle": # "circle" represents circling around the fire
                # Turn on the spot for the first 5 iterations
                if self.circle_counter > 7:
                    self.circle_turn90 = False
                print(self.circle_counter)
                
                # Run fire detection and P control
                movement, self.object_loc = self.followFire(self.img_bgr, self.circle_turn90) 
                self.circle_counter = self.circle_counter +1
                print("second")
                print(self.circle_counter)

                # Draws the line from the robot to the object
                cv2.line(self.img_bgr, [int(self.object_loc[0]), int(self.object_loc[1])], [int(self.robot_loc[0]), int(self.robot_loc[1])], (255,0,0), 1)
                cv2.circle(self.img_bgr, [int(self.object_loc[0]), int(self.object_loc[1])], 5, (255,0,0), -1)

                # Draws the progress bar for the circle counter
                cv2.line(self.img_bgr, [20,20], [20, int(20+100*1.4)], (0,0,0), 10)
                cv2.line(self.img_bgr, [20,20], [20, int(20+self.circle_counter*1.4)], (255,255,255), 10)

                if self.circle_counter > 1000: 
                    # If the robot has circled for 100 iterations, face the fire and reset
                    self.state = "search"
                    print("why?")
                    self.circle_counter = 0
                    self.circle_turn90 = True

            # Extract linear and angular movement commands
            if movement:
                self.current_linear = movement[0]
                self.current_angular = movement[1]
        except:
            pass    
        endtime = round((time.time() - start_time)*1000,2) # Stop the timer

        # Write info on the screen
        cv2.putText(self.img_bgr, self.state, (40,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(self.img_bgr, str(endtime)+" ms", (40,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(self.img_bgr, "speed: " + str(round(self.current_linear[0],2)), (40,120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(self.img_bgr, "turn: " + str(round(self.current_angular[2],2)), (40,160), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

        cv2.line(self.img_bgr, [int(self.robot_loc[0]), int(self.robot_loc[1])], [int(self.robot_loc[0]+self.robot_forw[0]*2), int(self.robot_loc[1]+self.robot_forw[1]*2)], (255,0,0), 1)

        # Show image
        cv2.imshow("camera", self.img_bgr)
        cv2.waitKey(1)
    
    ##### AR tag detection functions BEGIN #####
    def ar_detection(self, camera_feed):
        final_contour_list = self.contour_generator(camera_feed) # Generate contours to detect corners of the tag

        
        # If there are no conours, end and try again
        if len(final_contour_list) == 0:
            return [0,0], [0,0]
        
        # Creates an empty list
        morshu_list = list()
        dim = 200
        p1 = np.array([
            [0, 0],
            [dim - 1, 0],
            [dim - 1, dim - 1],
            [0, dim - 1]], dtype="float32")
        
        #print("Final contours:")
        # print(len(final_contour_list))
        # Go through all found contours
        for i in range(len(final_contour_list)):
            cv2.drawContours(self.img_bgr, [final_contour_list[i]], -1, (255, 0, 0), 2)
            # Draw all final contours and show the image
            # cv2.drawContours(camera_feed, [final_contour_list[i]], -1, (0, 255, 0), 2)
            # cv2.imshow("Outline", frame)
            # print(final_contour_list[i])
            # warped = homogenous_transform(small, final_contour_list[i][:, 0])

            # Seems to be extracting the corner coordinates of the contour
            c_rez = final_contour_list[i][:, 0]
            # print("Corners: " + str(c_rez))

            # Create homography matrix to convert from contour square to corrected perspective
            H_matrix = self.homograph(p1, self.order(c_rez))

            # H_matrix = homo(p1,order(c))
            # Perspective correct the ar tag using the homography matrix
            tag = cv2.warpPerspective(camera_feed, H_matrix, (200, 200))

            

            # Create a b/w copy of the tag
            tag1 = cv2.cvtColor(tag, cv2.COLOR_BGR2GRAY)

            # Get tag id and location of orientation square
            decoded, location = self.id_decode(tag1)
            

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
                #print("Location: " + location)
                cent, forw = self.tag_location(location, self.order(c_rez))

                # cv2.circle(camera_feed, [int(cent[0]),int(cent[1])], 3, (0,0,255), 2)
                # cv2.line(camera_feed, [int(cent[0]),int(cent[1])], [int(cent[0]+forw[0]), int(cent[1]+forw[1])], (0,0,255), 2)
                #print(cent)
                #print(forw)
                return cent, forw

            else:
                print("Location: NONE")


    def contour_generator(self, frame):
        # Create b/w of the video feed
        test_img1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Blur image for cleaner edges
        test_blur = cv2.GaussianBlur(test_img1, (5, 5), 0)
        # Edge detection
        edge = cv2.Canny(test_blur, 75, 200)
        # Make copy of edge image
        edge1 = copy.copy(edge)

        # Create empty list
        contour_list = list()

        # Find contours of the edge image
        cnts, h = cv2.findContours(edge1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # |   \___ Hieracy
        #  \______ Contours
        
        # This part crashes if there are no contours
        try:
            # Create empty list of indexes
            index = list()
            # Loop through all hieracies and save the ones with a hieracy higher than -1 (those whith parents)
            for hier in h[0]:
                if hier[3] != -1:
                    index.append(hier[3])

            # loop over the contours that pass the hieracy test
            for c in index:
                # Calculate the length of the edges of a contour (maybe)
                peri = cv2.arcLength(cnts[c], True)
                # Something about the amount of edges of the contour
                approx = cv2.approxPolyDP(cnts[c], 0.02 * peri, True)

                # If a contour has more than 4 edges
                if len(approx) > 4:
                    # some checks that make sure the contour is good
                    peri1 = cv2.arcLength(cnts[c - 1], True)
                    corners = cv2.approxPolyDP(cnts[c - 1], 0.02 * peri1, True)
                    # add the contour to the list
                    contour_list.append(corners)

            # Create NEW empty list
            new_contour_list = list()
            # Only keep those contours with length four
            for contour in contour_list:
                if len(contour) == 4:
                    new_contour_list.append(contour)
            
            # Create *FINAL* empty list
            final_contour_list = list()
            # Only keep those contours with an area of less than 2500
            for element in new_contour_list:
                if cv2.contourArea(element) < 3000 and cv2.contourArea(element) > 100:
                    # print("Area: " + str(cv2.contourArea(element)))
                    final_contour_list.append(element)

            return final_contour_list
        except:
            return 0

    def order(self, pts):
        rect = np.zeros((4, 2), dtype="float32")

        s = pts.sum(axis=1)
        # print(np.argmax(s))
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        # print(np.argmax(diff))
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        # return the ordered coordinates
        return rect

    def homograph(self, p, p1):
        A = []
        p2 = self.order(p)

        for i in range(0, len(p1)):
            x, y = p1[i][0], p1[i][1]
            u, v = p2[i][0], p2[i][1]
            A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
            A.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
        A = np.array(A)
        U, S, Vh = np.linalg.svd(A)
        l = Vh[-1, :] / Vh[-1, -1]
        h = np.reshape(l, (3, 3))
        # print(l)
        # print(h)
        return h

    def id_decode(self, image):
        # create binary from the input image
        ret, img_bw = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)

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

        # Not sure what this does, but it is used when finding the orientation
        corner_pixel = 255

        # Crop to isolate the center square
        cropped_img = img_bw[50:150, 50:150]

        # Select center pixels of the four id blocks
        block_1 = cropped_img[37, 37]
        block_3 = cropped_img[62, 37]
        block_2 = cropped_img[37, 62]
        block_4 = cropped_img[62, 62]

        # In case you wondered what white is:
        white = 255

        # strange way of identifying whether a block is white or not
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
        if cropped_img[85, 87] == corner_pixel:
            return list([block_3, block_4, block_2, block_1]), "BL"
        elif cropped_img[12, 87] == corner_pixel:
            return list([block_4, block_2, block_1, block_3]), "BR"
        elif cropped_img[12, 12] == corner_pixel:
            return list([block_2, block_1, block_3, block_4]), "TR"
        elif cropped_img[87, 12] == corner_pixel:
            return list([block_1, block_3, block_4, block_2]), "TL"

        return None, None

    def tag_location(self, orient, corners):
        # creates a new one of those corner things used for homography, but rotates it based on the orientation
        front_loc = [0,0]
        if orient == "TR":
            front_loc = [(corners[0][0]+corners[3][0])/2, (corners[0][1]+corners[3][1])/2]
        elif orient == "TL":
            front_loc = [(corners[2][0]+corners[3][0])/2, (corners[2][1]+corners[3][1])/2]
        elif orient == "BL":
            front_loc = [(corners[1][0]+corners[2][0])/2, (corners[1][1]+corners[2][1])/2]
        elif orient == "BR":
            front_loc = [(corners[0][0]+corners[1][0])/2, (corners[0][1]+corners[1][1])/2]
        
        center_loc = [(corners[0][0]+corners[1][0]+corners[2][0]+corners[3][0])/4, (corners[0][1]+corners[1][1]+corners[2][1]+corners[3][1])/4]
        forward_vector = [front_loc[0]-center_loc[0], front_loc[1]-center_loc[1]]
        return center_loc, forward_vector
    #####  AR tag detection functions END  #####

    # Calculate angle between two vectors
    def find_angle(self, sVector, eVector):
        # Components of the vectors
        a = sVector[0]
        b = sVector[1]
        c = eVector[0]
        d = eVector[1]

        dotProduct = a*c + b*d # take the dot product of the vectors
        
        modOfVector1 = math.sqrt( a*a + b*b)*math.sqrt(c*c + d*d) # Pythagoras theorem

        # Handles the zero case
        if modOfVector1 == 0:
            modOfVector1 = 0.01

        #angle = dotProduct/modOfVector1 # calculate cos of the angle

        angle = math.atan2( a*d - b*c, a*c + b*d )

        angleInDegree = math.degrees(angle) # Convert to degrees

        # invert angle when vector one is to the left of vector two (or maybe the other way)
        # if a < c:
        #     angleInDegree = angleInDegree*-1

        return angleInDegree

    def tangent_angle(self, object, dist, target):
        rob_obj = object - self.robot_loc
        angle = -self.find_angle(self.robot_forw, rob_obj) +90

        err = target - dist

        heading = -(err + angle)

        print("Angle:   " + str(angle))
        print("Err:     " + str(err))
        print("heading: " + str(heading))

        return heading

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
        img_blur = cv2.GaussianBlur(camera_feed, (3, 3), 0)
        image = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HLS) # Convert image to HLS
        
        kernel = np.ones((9, 9), np.uint8) # Used in noise reduction
        
        threshold = 100 # Change this depending on what you need it as
        nearest_point = [0,0] # Variable for storing the neares object pixel
        center = np.array((image.shape[0] / 2, image.shape[1] / 2))  # Calculates the center of the image

        # P controller for the linear velocity, based on the distance to the object
        pidVel = PID(search_vel_pid[0],search_vel_pid[1],search_vel_pid[2],setpoint = 100)
        pidVel.output_limits = (search_vel_lim[0],search_vel_lim[1])

        # P controller for the angular velocity, based on the angle between the robot and the object
        pidAng = PID(search_ang_pid[0],search_ang_pid[1],search_ang_pid[2], setpoint = 0)
        pidAng.output_limits = (search_ang_lim[0],search_ang_lim[1])

        # Main
        (h, l, s) = cv2.split(image) # Split the channels of the image and save the hue channel
        image = cv2.inRange(h, 85, 95) # threshold the image to find the fire
        image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel) # Noise reduction (Remove this later perhaps)

        cv2.imshow("something",image)

        shortest_dist = 1000000 # arbitrarily high starting value for shortest distance to object
        nonzero = cv2.findNonZero(image) # Find locations of all non-zero pixels in image

        nearest_point = [0,0]
        # If there is no object it gets angry, so I did a try/except to handle it
        try:
            distances = np.sqrt((nonzero[:,:,0] - self.robot_loc[0]) ** 2 + (nonzero[:,:,1] - self.robot_loc[1]) ** 2) # Measure distances to all nonzero entries
            nearest_index = np.argmin(distances) # Find the index of the closest pixel
            nearest_point = nonzero[nearest_index] # Get coordinated of nearest pixel
            shortest_dist = np.sqrt((nearest_point[0][0] - self.robot_loc[0]) ** 2 + (nearest_point[0][1] - self.robot_loc[1]) ** 2) # Calculate shortest distance... again
        except:
            # If no object is present, set nearest pixel to the center pixel
            nearest_point = [[image.shape[0] / 2, image.shape[1] / 2]]
            shortest_dist = 0
        
        # Reset turn and speed before calculating new values
        turn = 0
        targetSpeed = 0
        
        # Calculate the vectors for robot to object, and robot forward
        rob_obj = [nearest_point[0][0]-self.robot_loc[0], nearest_point[0][1]-self.robot_loc[1]]
        # rob_for = [0,-100]

        # self.img_bgr = cv2.line(self.img_bgr, (int(rob_obj[0]+self.robot_loc[0]),int(rob_obj[1]+self.robot_loc[1])), (int(rob_for[0]+self.robot_loc[0]),int(rob_for[1]+self.robot_loc[1])), (255,0,0), 1)

        object_angle = self.find_angle(rob_obj, self.robot_forw) # Calculate angle to the object

        turn = pidAng(-object_angle) # Set turn value based on the x-coordinate of the nearest point
            
        if shortest_dist > threshold: # Check if the object is within the desired distance
            targetSpeed = -pidVel(shortest_dist) # Set speed value based on the distance to the nearest point
        elif abs(turn) < 0.05: # I the object is within acceptable distance and turn value is suitably small
            self.state = "circle" # Change the state of the robot to "circle"

        return [[targetSpeed,0,0], [0,0,-turn]], nearest_point[0] # Return the twist msg components 

    # Circle around fire using P-controller 
    def followFire(self, camera_feed, turn_state):
        
        # Create PID cotroller with a target value that is the same as the distance for the approach
        followPID = PID(circle_ang_pid[0],circle_ang_pid[1],circle_ang_pid[2], setpoint=0)
        
        followPID.output_limits = (circle_ang_lim[0],circle_ang_lim[1])
        
        kernel = np.ones((9, 9), np.uint8) # Kernel used in noise reduciton
        
        # If turn state is true then turn left (five iterations gives approximately 90 degrees)
        if turn_state:
            print("Turning!")
            return [[-0.1,0,0], [0,0,-1]], self.robot_loc
        
        img_blur = cv2.GaussianBlur(camera_feed, (3, 3), 0)
        image = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HLS) # Convert image to HLS
        (h, l, s) = cv2.split(image) # Split channels and save hue channel 
        image = cv2.inRange(h, 85, 95) # Threshhold image for objects
        image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)         # Noise reduction (Remove this later perhaps)

        # shortest_path = 1000000 # Arbitrarily high shortest path value
        center = self.robot_loc
        # np.array((image.shape[0] / 2, image.shape[1] / 2))  # Calculates the center of the image
        
        nonzero = cv2.findNonZero(image) # Find all non-zero pixels in the image
        
        try:
            distances = np.sqrt((nonzero[:,:,0] - center[0]) ** 2 + (nonzero[:,:,1] - center[1]) ** 2) # Measure distances to all nonzero entries
            nearest_index = np.argmin(distances) # Find the index of the closest pixel
            nearest_point = nonzero[nearest_index] # Get coordinated of nearest pixel
            # self.object_loc = nearest_point
            shortest_dist = np.sqrt((nearest_point[0][0] - center[0]) ** 2 + (nearest_point[0][1] - center[1]) ** 2) # Calculate shortest distance... again
        except:
            # If no object is present, set nearest pixel to the center pixel
            nearest_point = [[image.shape[0] / 2, image.shape[1] / 2]]
            # self.object_loc = nearest_point
            shortest_dist = 100
        
        print("Dist: " + str(shortest_dist))
        # If the distance is too large, return to search
        # if shortest_dist > 200:
        #     print("Panic switch")
        #     self.state = "search"
        #     self.circle_turn90 = True
        #     self.circle_counter = 0
        #     return [[0.1,0,0], [0,0,0]], nearest_point[0]

        target_angle = self.tangent_angle(nearest_point[0], shortest_dist, 100)

        turn_pid = followPID(target_angle) # Calculate turn value based on distance to object

        return [[0.2,0,0], [0,0,turn_pid]], nearest_point[0] # Return twist msg components
        
    # Runs every set interval and sends twist message
    def on_tmr(self):
        success, frame = self.cap.read()

        if success:
            self.image_callback(frame)


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