# Imports
import numpy as np
import cv2
from matplotlib import pyplot as plt
import copy
# import imutils
import math
import time

# Initiate video capture
cap = cv2.VideoCapture(0)

# Import morshu image and resize it
morshu_img = cv2.imread('Controllers/AR-tag/Morshu.png')
morshu_resize = cv2.resize(morshu_img, (200, 200))

start_time = 0

# Not sure what this is, seems to describe the corners of a square
dim = 200
p1 = np.array([
    [0, 0],
    [dim - 1, 0],
    [dim - 1, dim - 1],
    [0, dim - 1]], dtype="float32")

# Decode the tag ID and 4 digit binary and orientation
def id_decode(image):
    # create binary from the input image
    ret, img_bw = cv2.threshold(image, 180, 255, cv2.THRESH_BINARY)

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
        return list([block_3, block_4, block_2, block_1]), "BR"
    elif cropped_img[12, 87] == corner_pixel:
        return list([block_4, block_2, block_1, block_3]), "TR"
    elif cropped_img[12, 12] == corner_pixel:
        return list([block_2, block_1, block_3, block_4]), "TL"
    elif cropped_img[87, 12] == corner_pixel:
        return list([block_1, block_3, block_4, block_2]), "BL"

    return None, None

# Function to return the order of points in camera frame
def order(pts):
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

# Function to compute homography between world and camera frame 
def homograph(p, p1):
    A = []
    p2 = order(p)

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

# Generate contours to detect corners of the tag
def contour_generator(frame):
    # Create b/w of the video feed
    test_img1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Blur image for cleaner edges
    test_blur = cv2.GaussianBlur(test_img1, (5, 5), 0)
    # Edge detection
    edge = cv2.Canny(test_blur, 75, 200) #75,200
    cv2.imshow("edges", edge)
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
                print("Area: " + str(cv2.contourArea(element)))
                final_contour_list.append(element)

        return final_contour_list
    except:
        return 0
    
# Reorient the tag based on the original orientation
# def reorient(location, maxDim):
#     # creates a new one of those corner things used for homography, but rotates it based on the orientation
#     if location == "BR":
#         p1 = np.array([
#             [0, 0],
#             [maxDim - 1, 0],
#             [maxDim - 1, maxDim - 1],
#             [0, maxDim - 1]], dtype="float32")
#         return p1
#     elif location == "TR":
#         p1 = np.array([
#             [maxDim - 1, 0],
#             [maxDim - 1, maxDim - 1],
#             [0, maxDim - 1],
#             [0, 0]], dtype="float32")
#         return p1
#     elif location == "TL":
#         p1 = np.array([
#             [maxDim - 1, maxDim - 1],
#             [0, maxDim - 1],
#             [0, 0],
#             [maxDim - 1, 0]], dtype="float32")
#         return p1

#     elif location == "BL":
#         p1 = np.array([
#             [0, maxDim - 1],
#             [0, 0],
#             [maxDim - 1, 0],
#             [maxDim - 1, maxDim - 1]], dtype="float32")
#         return p1

# Genious function for finding the location and direction of the ar tag
def tag_location(orient, corners):
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

# Main function to process the tag
#      video feed --v    v-- corner thingy 
def image_process(frame, p1):
    final_contour_list = contour_generator(frame) # Generate contours to detect corners of the tag

    # If there are no conours, end and try again
    if final_contour_list == 0:
        return
    
    # Creates an empty list
    morshu_list = list()

    # print("Final contours:")
    print(len(final_contour_list))
    # Go through all found contours
    for i in range(len(final_contour_list)):
        # Draw all final contours and show the image
        cv2.drawContours(frame, [final_contour_list[i]], -1, (0, 255, 0), 2)
        # cv2.imshow("Outline", frame)
        # print(final_contour_list[i])
        # warped = homogenous_transform(small, final_contour_list[i][:, 0])

        # Seems to be extracting the corner coordinates of the contour
        c_rez = final_contour_list[i][:, 0]
        # print("Corners: " + str(c_rez))

        # Create homography matrix to convert from contour square to corrected perspective
        H_matrix = homograph(p1, order(c_rez))

        # H_matrix = homo(p1,order(c))
        # Perspective correct the ar tag using the homography matrix
        tag = cv2.warpPerspective(frame, H_matrix, (200, 200))

        

        # Create a b/w copy of the tag
        tag1 = cv2.cvtColor(tag, cv2.COLOR_BGR2GRAY)

        # Get tag id and location of orientation square
        decoded, location = id_decode(tag1)
        

        cv2.circle(tag, [62,62], 1, (255,0,0), 2)
        cv2.circle(tag, [62,137], 1, (255,0,0), 2)
        cv2.circle(tag, [137,62], 1, (255,0,0), 2)
        cv2.circle(tag, [137,137], 1, (255,0,0), 2)

        cv2.circle(tag, [87,87], 1, (0,0,255), 2)
        cv2.circle(tag, [87,112], 1, (0,0,255), 2)
        cv2.circle(tag, [112,87], 1, (0,0,255), 2)
        cv2.circle(tag, [112,112], 1, (0,0,255), 2)
        

        # Create an empty image frame of same size as video frame
        # empty = np.full(frame.shape, 0, dtype='uint8')

        # If a location has been gathered
        if not location == None:
            print("Location: " + location)
            cent, forw = tag_location(location, order(c_rez))

            cv2.circle(frame, [int(cent[0]),int(cent[1])], 2, (0,0,255), 2)
            cv2.line(frame, [int(cent[0]),int(cent[1])], [int(cent[0]+forw[0]), int(cent[1]+forw[1])], (0,0,255), 2)
            

        #     # Get new corner description thing, with corners stored clockwise
        #     p2 = reorient(location, 200)
        #     # if not decoded == None:
        #     #     print("ID detected: " + str(decoded))
        #     # Create homography the other direction, from corrected tag to original image
        #     H_morshu = homograph(order(c_rez), p2)

        #     # Create perspective "uncorrected" morshu
        #     morshu_overlap = cv2.warpPerspective(morshu_resize, H_morshu, (frame.shape[1], frame.shape[0]))

        #     # Append the morshu image to the morshu list for later use
        #     if not np.array_equal(morshu_overlap, empty):
        #         morshu_list.append(morshu_overlap.copy())
        #         # print(morshu_overlap.shape)
        else:
            print("Location: NONE")
        
        # Draw the forward line
        

        # Display found contours and perspective corrected tag
        cv2.imshow("Outline", frame)
        cv2.imshow("Tag after homography", tag)

    # create and empty image that will be used as a mask when adding morshu to the tag
    # mask = np.full(frame.shape, 0, dtype='uint8')

    # # If there are morshus in the list
    # if morshu_list != []:
    #     # add all morshus to the mask
    #     for morshu in morshu_list:
    #         temp = cv2.add(mask, morshu.copy())
    #         mask = temp

    #     # Create a binary of where all morshus are (don't know why the threshold cuts off at 10 though)
    #     morshu_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    #     r, morshu_bin = cv2.threshold(morshu_gray, 10, 255, cv2.THRESH_BINARY)

    #     # Invert the mask, so that all areas where there isn't a morshu are white
    #     mask_inv = cv2.bitwise_not(morshu_bin)


    #     mask_3d = frame.copy()
    #     mask_3d[:, :, 0] = mask_inv
    #     mask_3d[:, :, 1] = mask_inv
    #     mask_3d[:, :, 2] = mask_inv

    #     # Bitwise and to create a version of the original frame, where the tag has been blacked out
    #     img_masked = cv2.bitwise_and(frame, mask_3d)
        
    #     # Add the pespective "uncorrected" morshu to the black part of the image
    #     final_image = cv2.add(img_masked, mask)
        
    #     # cv2.putText(frame, str(int((time.time()-start_time)*1000)), [20,20], cv2.FONT_HERSHEY_SIMPLEX, 1, [0,0,0], 2, cv2.LINE_AA)
        
    #     # Display the final image
    #     cv2.imshow("Morhsu", final_image)
    #     # cv2.waitKey(0)
    
    if cv2.waitKey(1) & 0xff == 27:
        cv2.destroyAllWindows()

# Read the input video frame by frame
while True:
    start_time = time.time() # Start timer

    success, frame = cap.read() # Captures the image from the camera

    img = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5) # Resize the camera image

    image_process(img, p1) # Function that does all of the IP
    
    print("Process time: " + str((time.time()-start_time)*1000))

    if cv2.waitKey(20) & 0xFF == ord(' '): #breaks the loop when space is pressed
        break
