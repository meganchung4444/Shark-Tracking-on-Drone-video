# Import python libraries
import numpy as np
import cv2

def detect(frame):
    # Convert BGR to GRAY
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect edges
    edges = cv2.Canny(gray,  50, 190, 3)

    # Retain only edges within the threshold
    _, thresh = cv2.threshold(edges, 127, 255,0)#cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, \
                                    cv2.CHAIN_APPROX_SIMPLE)

    # Set the accepted minimum & maximum radius of a detected object
    min_radius_thresh= 2
    max_radius_thresh= 40
    
    # ref: https://docs.opencv.org/trunk/dd/d49/tutorial_py_contour_features.html
    centers=[]
    for c in contours:        
        (x, y), radius = cv2.minEnclosingCircle(c)
        center = (int(x), int(y))
        radius = int(radius)
        
        if (radius > min_radius_thresh) and (radius < max_radius_thresh):
            radius = 5  # Set the same radius to draw  detected circles
            cv2.circle(frame, center, radius, (0, 0, 0), 2)            
            centers.append((x,y))
    return np.array(centers)