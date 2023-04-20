import cv2
import numpy as np

img_bgr = cv2.imread('src/articubot_one/test_frame.png', cv2.IMREAD_COLOR)

img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

h,s,c = cv2.split(img_hsv)

n=0
mask = cv2.inRange(h, 1, 20)
mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

M = cv2.moments(mask)
 
# calculate x,y coordinate of center
cX = int(M["m10"] / M["m00"])
cY = int(M["m01"] / M["m00"])

cv2.circle(mask_bgr, (cX, cY), 5, (0,0,255), -1)

cv2.imshow("image", mask_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()