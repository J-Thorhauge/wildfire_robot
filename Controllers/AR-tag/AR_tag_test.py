import numpy
import cv2 as cv

cap = cv. VideoCapture(0)

while True:
    success, img = cap.read() #captures the image from the camera
    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    blur = cv.GaussianBlur(imgGray, (3,3), cv.BORDER_DEFAULT)
    ret, thresh = cv.threshold(blur, 150, 255, 0)
    contours, hierarchies = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    out = cv.cvtColor(blur, cv.COLOR_GRAY2BGR)
    for cnt in contours:
        if len(cnt)>100:
            #print(cnt)
            cv.drawContours(out, [cnt], 0, (0,0,255), 2)

    cv.imshow("Thresh", thresh) #displays the image
    cv.imshow("Hand tracker", out) #displays the image
    
    if cv.waitKey(20) & 0xFF == ord(' '): #breaks the loop when space is pressed
        break