from collections import defaultdict
import sys
import cv2
import numpy as np
import json
from ultralytics import YOLO
import supervision as sv
from fast_sam_ubuntu import FastSAMCustom

THICKNESS = 2
FONT_SCALE = 0.5
FONT = cv2.FONT_HERSHEY_SIMPLEX
COLOR = (255, 0, 0)
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

# Reference: https://stackoverflow.com/questions/40795709/checking-whether-two-rectangles-overlap-in-python-using-two-bottom-left-corners
def is_overlapping(rec1, rec2):
    # print("Testing overlapping between", rec1, "and", rec2)
    if (rec2['x2'] > rec1['x1'] and rec2['x2'] < rec1['x2']) or (rec2['x1'] > rec1['x1'] and rec2['x1'] < rec1['x2']):
        x_match = True
    else:
        x_match = False
    if (rec2['y2'] > rec1['y1'] and rec2['y2'] < rec1['y2']) or (rec2['y1'] > rec1['y1'] and rec2['y1'] < rec1['y2']):
        y_match = True
    else:
        y_match = False
    if x_match and y_match:
        return True
    else:
        return False

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

    def append_tracking_history(self, center):
        self.tracking_history.append(center)

    def draw_box(self, frame):
        # Draw Box
        cv2.rectangle(frame, (int(self.box["x1"]), int(self.box["y1"])), (int(self.box["x2"]), int(self.box["y2"])), (56, 56, 255), 2) 

    def draw_circle(self, frame):
        cv2.circle(frame, get_box_center(self.box), 5, LABEL_COLOR, 3)

    def draw_line(self, frame, other):
        cv2.line(frame, get_box_center(self.box), get_box_center(other.box), COLOR, thickness=THICKNESS, lineType=8)

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
        prev_objects = None

        if success:
            
            # Run YOLOv8 tracking on the frame, persisting tracks between frames
            results = model(frame)
            
            """
            # Custom SAM
            sam_results = fs.get_json_data(frame=frame)
            if sam_results:
                for idx, r in enumerate(sam_results):
                    # print(f"SAM {frame_cnt}-{idx}", r)
                    continue
            else:
                print("No SAM results")
                # Doc: https://docs.ultralytics.com/modes/predict/#boxes
            """

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
            for objects in frame_tracker:
                
                if len(objects) > 0:

                    for obj in objects: 

                        # Get the box center
                        center = get_box_center(obj.box)
                        
                        if not prev_objects:
                            
                            # Draw a circle
                            cv2.circle(frame, center, 5, LABEL_COLOR, 3)
                        
                            # Draw a label
                            obj.draw_label(frame, f"Shark at: t{obj.frame_cnt}")
        
                        # Draw a line 
                        if prev_objects:
                            
                            # Connect dectected objects from previous frame and current frame if any of them are overlapping
                            for prev_obj in prev_objects:
                                prev_obj.draw_line(frame, obj)
                         
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
