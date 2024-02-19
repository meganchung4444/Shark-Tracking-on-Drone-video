# Import libraries
import numpy as np
import cv2

def detect(frame):
    # Convert frame from BGR to GRAY
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
    # Edge detection using Canny function
    img_edges = cv2.Canny(gray,  50, 190)
    
    # Convert to black and white image
    _ , img_thresh = cv2.threshold(img_edges, 254, 255,cv2.THRESH_BINARY)            

    # Find contours
    contours, _ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)    

    # Set the minimum & maximum radius of the accepted circle
    min_radius_thresh= 3
    max_radius_thresh= 30

    centers=[]
    for c in contours:
        # ref: https://docs.opencv.org/trunk/dd/d49/tutorial_py_contour_features.html
        (x, y), radius = cv2.minEnclosingCircle(c)
        radius = int(radius)

        #Get only the valid circle(s)
        if (radius > min_radius_thresh) and (radius < max_radius_thresh):
            #centers.append(np.array([[x], [y]]))
            centers.append((x,y))
    return np.array(centers)