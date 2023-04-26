import cv2
import numpy as np

# img_bgr = cv2.imread('src/articubot_one/test_frame.png', cv2.IMREAD_COLOR)
cap = cv2.VideoCapture(0)

# 0 skin - 1 bilka bag - 2 normal bag
obj_id = 2
thresh = [[0,18],[70,110],[85,95]]

while True:
    success, frame = cap.read()

    img_blur = cv2.GaussianBlur(frame, (3, 3), 0)

    img_hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)

    h,s,c = cv2.split(img_hsv)

    n=0
    
    mask = cv2.inRange(h, thresh[obj_id][0], thresh[obj_id][1])

    kernel = np.ones((7,7),np.uint8)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    mask_bgr = cv2.cvtColor(opening, cv2.COLOR_GRAY2BGR)

    M = cv2.moments(opening)
    
    # calculate x,y coordinate of center
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    cv2.circle(mask_bgr, (cX, cY), 5, (0,0,255), -1)

    cv2.imshow("image", mask_bgr)
    if cv2.waitKey(20) & 0xFF == ord(' '): #breaks the loop when space is pressed
        break