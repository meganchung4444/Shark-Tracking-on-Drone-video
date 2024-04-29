import cv2
import seaborn as sns
import numpy as np

THICKNESS = 2
iFONT_SCALE = 0.7
FONT = cv2.FONT_HERSHEY_SIMPLEX
COLOR = (10, 20, 30)
INTERPOLATION_COLOR = (38, 146, 240)
LABEL_COLOR = (56, 56, 255)
color_list = sns.color_palette('bright', 20) 
color_list = [(int(r*255), int(g*255), int(b*255)) for (r, g, b) in color_list]

def get_box_center(box):
    x1 = box["x1"]
    x2 = box["x2"]
    y1 = box["y1"]
    y2 = box["y2"]
    return np.array([int(x1 + ((x2 - x1)//2)), int(y1 + ((y2 - y1)//2))])

# Reference: https://stackoverflow.com/questions/40795709/checking-whether-two-rectangles-overlap-in-python-using-two-bottom-left-corners
def is_overlapping(rect1, rect2):
    
    x1 = rect1["x1"]
    y1 = rect1["y1"]
    w1 = rect1["x2"] - rect1["x1"]
    h1 = rect1["y2"] - rect1["y1"]

    x2 = rect2["x1"]
    y2 = rect2["y1"]
    w2 = rect2["x2"] - rect2["x1"]
    h2 = rect2["y2"] - rect2["y1"]
    
    # Calculate the coordinates of the top-left and bottom-right corners of each rectangle
    top_left1 = (x1, y1)
    bottom_right1 = (x1 + w1, y1 + h1)
    top_left2 = (x2, y2)
    bottom_right2 = (x2 + w2, y2 + h2)

    # Check for intersection
    if (top_left1[0] < bottom_right2[0] and
        bottom_right1[0] > top_left2[0] and
        top_left1[1] < bottom_right2[1] and
        bottom_right1[1] > top_left2[1]):
        return True  # Rectangles are intersecting
    else:
        return False  # Rectangles are not intersecting
    

class GeneralObject():

    def __init__(self, name, cls, box, confidence, frame_cnt):
        self.name = name
        self.cls = cls
        self.box = box
        self.confidence = confidence
        self.frame_cnt = frame_cnt
        self.center = get_box_center(self.box)

    def __str__(self):
        return f"[ Name={self.name} Box={self.box} ]"

    def as_dict(self):
        return {"name": self.name, "class":self.cls, "box":self.box, "confidence":self.confidence, "frame_cnt":self.frame_cnt}

    def update_box(self, box):
        self.box = box
        self.center = get_box_center(box)

    def update_confidence(self, confidence):
        self.confidence = confidence

    def update_frame_cnt(self, frame_cnt):
        self.frame_cnt = frame_cnt

    def draw_box(self, frame, color = LABEL_COLOR):
        cv2.rectangle(frame, (int(self.box["x1"]), int(self.box["y1"])), (int(self.box["x2"]), int(self.box["y2"])), color, 2) 

    # def draw_circle(self, frame):
    #     cv2.circle(frame, get_box_center(self.box), 5, LABEL_COLOR, 3)

    def draw_line(self, frame, other, color = COLOR, thickness = THICKNESS):
        c1, c2, c3 = color
        #c1 = (c1 + self.frame_cnt % 255)
        #c2 = (c2 + self.frame_cnt % 255)
        #c3 = (c3 + self.frame_cnt % 255)
        cv2.line(frame, self.center, other.center, (c1, c2, c3), thickness=thickness, lineType=8)

    def draw_label(self, frame, text, color = LABEL_COLOR):
        label_pos = (self.center[0], self.center[1]-2)
        cv2.putText(frame, text, label_pos, FONT,  
                   FONT_SCALE, color, THICKNESS, cv2.LINE_AA)

    def __eq__(self, other):
        return is_overlapping(self.box, other.box)