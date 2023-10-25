from collections import defaultdict
import sys
import cv2
import numpy as np
import json
from ultralytics import YOLO
import supervision as sv

THICKNESS = 2
FONT_SCALE = 0.7
FONT = cv2.FONT_HERSHEY_SIMPLEX
COLOR = (10, 20, 30)
LABEL_COLOR = (56, 56, 255)

def load_video(video_path, output_path="./results/output.mp4"):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    return cap, out

def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)

def display_frame(annotated_frame):
        
    # Display the annotated frame
    resize = ResizeWithAspectRatio(annotated_frame, width=1200, height=800)
    cv2.imshow("YOLOv8 Tracking", resize)

    # Resizing the window
    cv2.resizeWindow("YOLOv8 Tracking", 1200, 800)


def is_shark_missed(json_format):
    missed = True
    for obj in json_format:
        if obj["name"] == "shark":
            missed = False
    return missed

# def calculate_xywh()

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
    
def find_sharks_by_sam(prev_sharks_prediction_list, sam_results):
    results = []
    for shark in prev_sharks_prediction_list:
        shark_name, shark_cls, shark_box, shark_confidence = shark
        for obj in sam_results:
            obj_name, obj_cls, obj_box, obj_confidence = obj
            if is_overlapping(shark_box, obj_box):
                results.append(('shark', obj_cls, obj_box, obj_confidence))
    return results

def get_box_center(box):
    x1 = box["x1"]
    x2 = box["x2"]
    y1 = box["y1"]
    y2 = box["y2"]
    return (int(x1 + ((x2 - x1)//2)), int(y1 + ((y2 - y1)//2)))

class GeneralObject():

    def __init__(self, name, cls, box, confidence, frame_cnt):
        self.name = name
        self.cls = cls
        self.box = box
        self.tracking_history = []
        self.confidence = confidence
        self.frame_cnt = frame_cnt

    def __str__(self):
        return f"[ Name={self.name} Box={self.box} ]"

    def update_box(self, box):
        self.box = box

    def update_confidence(self, confidence):
        self.confidence = confidence

    def update_frame_cnt(self, frame_cnt):
        self.frame_cnt = frame_cnt

    def draw_box(self, frame):
        cv2.rectangle(frame, (int(self.box["x1"]), int(self.box["y1"])), (int(self.box["x2"]), int(self.box["y2"])), (56, 56, 255), 2) 

    def draw_circle(self, frame):
        cv2.circle(frame, get_box_center(self.box), 5, LABEL_COLOR, 3)

    def draw_line(self, frame, other):
        c1, c2, c3 = COLOR
        #c1 = (c1 + self.frame_cnt % 255)
        #c2 = (c2 + self.frame_cnt % 255)
        #c3 = (c3 + self.frame_cnt % 255)
        cv2.line(frame, get_box_center(self.box), get_box_center(other.box), (c1, c2, c3), thickness=THICKNESS, lineType=8)

    def draw_label(self, frame, text):
        center = get_box_center(self.box)
        label_pos = (center[0], center[1]-2)
        cv2.putText(frame, text, label_pos, FONT,  
                   FONT_SCALE, LABEL_COLOR, THICKNESS, cv2.LINE_AA)

    def __eq__(self, other):
        return is_overlapping(self.box, other.box)


def main(model_path="best.pt", video_path="./assets/example_vid_1.mp4", output_path="./results/test.mp4", standard_confidence=0.77):   

    frame_tracker = []

    # 1. Set up a model
    model = YOLO(model_path)
    model.to("cuda")

    # 2. Video Setup
    # Open the video file
    cap, video_writer = load_video(video_path, output_path)

    # fs = FastSAMCustom()

    # 3. SAM setup
    # Set up SAM

    # Loop through the video frames
    frame_cnt = 1
    while cap.isOpened():

        # Read a frame from the video
        success, frame = cap.read()
        frame_tracker.append([])
        
        if success:
            
            # Run YOLOv8 tracking on the frame, persisting tracks between frames
            results = model(frame)
            

            # 1. Iterate the YOLO results
            for idx, r in enumerate(results):
                # Returns torch.Tensor(https://pytorch.org/docs/stable/tensors.html)
                # xywh returns the boxes in xywh format.
                # cpu() moves the object into cpu
                boxes = r.boxes.xywh.cpu()

                
                # Contains box plot information
                yolo_json_format = json.loads(r.tojson())

                # 1-1. Construct all object list and shark list
                for obj in yolo_json_format:     
                    name = obj["name"]
                    cls = obj["class"]
                    box = obj["box"]
                    confidence = obj["confidence"]
                                            
                    # Create a new General Object
                    new_obj = GeneralObject(name, cls, box, confidence, frame_cnt)
                    
                    # Append the object if it has a high possiblity of being a shark
                    if new_obj.name == 'shark' and new_obj.confidence > standard_confidence:
                        frame_tracker[frame_cnt-1].append(new_obj)

                
            # 2. Draw Tracking History for each frame
            prev_objects = []
            recently_detected_objects = []
            for objects in frame_tracker:

                for i, obj in enumerate(objects):
    
                    # Get the box center
                    center = get_box_center(obj.box)
        
                    # Plot the object's location on the frame
                    
                    if len(objects) > 0 and len(prev_objects) == 0:
                        
                        # Draw a circle
                        cv2.circle(frame, center, 2, LABEL_COLOR, 3)
                
                        # Draw a label
                        obj.draw_label(frame, f"t{obj.frame_cnt}")  
                    
                
                    # if prev_objects is None:
                        # cv2.circle(frame, center, 2, LABEL_COLOR, 3)
                    
                    # Connect lines if there are previously detected objects
                    if len(prev_objects) > 0:
                        
                        # Check the box detection
                        for prev_obj in prev_objects:
                            if prev_obj == obj:
                                prev_obj.draw_line(frame, obj)
                                obj.update_frame_cnt(prev_obj.frame_cnt)

                                 
                    
                """
                    # Connect if there is 
                    if len(objects) > 0 and len(prev_objects) == 0:
                        for recent_obj in recently_detected_objects:
                            recent_obj.draw_line(frame, obj)
                            
                if len(objects) == 0 and len(prev_objects) > 0:
                    recently_detected_objects = prev_objects
                """        


                # Update prev_objects
                prev_objects = objects
                
            
            # 3. Write into the video file & increase the frame counter
            video_writer.write(frame)
            frame_cnt+=1

        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    video_writer.release()
    cap.release()

if __name__ == "__main__":
    # Check if the user provided the correct number of arguments
    if len(sys.argv) == 5:
                            
        # Get the command-line arguments
        model_path = sys.argv[1]
        video_path = sys.argv[2]
        output_path = sys.argv[3]
        standard_confidence = float(sys.argv[4])

        # Start the main function
        main(model_path, video_path, output_path, standard_confidence)
    
    elif len(sys.argv) == 1:
        main()

    else:

        # Print error
        print("Usage: python script.py model_path video_path output_path standard_confidence")
        sys.exit(1)
