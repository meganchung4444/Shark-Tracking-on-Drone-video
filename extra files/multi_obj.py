import sys
import cv2
import numpy as np
import json
from ultralytics import YOLO
import supervision as sv

"""
Author: Sunbeom (Ben) Kweon
Description: This program assumes that there is only one shark in the video
"""


THICKNESS = 2
iFONT_SCALE = 0.7
FONT = cv2.FONT_HERSHEY_SIMPLEX
COLOR = (10, 20, 30)
INTERPOLATION_COLOR = (38, 146, 240)
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

# Source: https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
def calc_iou(box1, box2):
    # find coordinates for intersection box
    # top left corner
    inner_x1 = max(box1["x1"], box2["x2"])
    inner_y1 = max(box1["y1"], box2["y1"])
    # bottom right corner
    inner_x2 = min(box1["x2"], box2["x2"])
    inner_y2 = min(box1["y2"], box2["y2"])

    # calc area of intersection box by multiplying width and height
    # if not intersect, area = 0
    # not sure about + 1
    intersection_area = max(0, inner_x2 - inner_x1) * max(0, inner_y2 - inner_y1)

    # calc the area of each box
    box1_area = (box1["x2"] - box1["x1"]) * (box1["y2"] - box1["y1"])
    box2_area = (box2["x2"] - box2["x1"]) * (box2["y2"] - box2["y1"]) 
    # subtract the intersection area to not consider it twice
    union_area = box1_area + box2_area - intersection_area

    return intersection_area / union_area # iou equation


def greater_iou(box1, box2, box3):
    # box1 would be detected box
    # box2 and box3 would be the 2 boxes that we are unsure is the next box in the path
    iou1 = calc_iou(box1, box2)
    iou2 = calc_iou(box1, box3)

    if iou1 > iou2:
        return box2
    return box3 

class GeneralObject():

    obj_id_counter = 1
    all_ids = set()
    id_frames_dict = {}

    def __init__(self, name, cls, box, confidence, frame_cnt, id):
        self.name = name
        self.cls = cls
        self.box = box
        self.confidence = confidence
        self.frame_cnt = frame_cnt
        self.center = get_box_center(self.box)
        self.id = id

    # def update_ids(self):
    #     if self.id not in GeneralObject.all_ids:
    #       GeneralObject.all_ids.add(self.id)

    # def set_id(self, frame_cnt, frames):
        
    #     # if it is the first frame, so no previous objects
    #     if len(frames) == 0:
    #         self.id = 1
    #         return
    #     prev_obj= frames[frame_cnt - 2]
    #     curr_obj = frames[frame_cnt - 1]
    #     if self.is_overlapping(prev_obj, curr_obj):
    #         self.id = prev_obj.id
    #         return
    #     else:
    #         self.id = GeneralObject.obj_id_counter + 1
        

    def __str__(self):
        return f"[ Name={self.name} Box={self.box} ]"

    def as_dict(self):
        return {"ID" : self.id, "name": self.name, "class":self.cls, "box":self.box, "confidence":self.confidence, "frame_cnt":self.frame_cnt}

    def update_box(self, box):
        self.box = box
        self.center = get_box_center(box)

    def update_confidence(self, confidence):
        self.confidence = confidence

    def update_frame_cnt(self, frame_cnt):
        self.frame_cnt = frame_cnt

    def draw_box(self, frame, color = LABEL_COLOR):
        cv2.rectangle(frame, (int(self.box["x1"]), int(self.box["y1"])), (int(self.box["x2"]), int(self.box["y2"])), color, 2) 
        cv2.putText(frame, str(self.id), (int(self.box["x1"]), int(self.box["y1"]) - 5), FONT, iFONT_SCALE, color, THICKNESS, cv2.LINE_AA)

    def draw_circle(self, frame):
        cv2.circle(frame, get_box_center(self.box), 5, LABEL_COLOR, 3)

    def draw_line(self, frame, other, color = COLOR, thickness = THICKNESS):
        c1, c2, c3 = color
        #c1 = (c1 + self.frame_cnt % 255)
        #c2 = (c2 + self.frame_cnt % 255)
        #c3 = (c3 + self.frame_cnt % 255)
        cv2.line(frame, self.center, other.center, (c1, c2, c3), thickness=thickness, lineType=8)

    def draw_label(self, frame, text, color = LABEL_COLOR):
        label_pos = (self.center[0], self.center[1]-2)
        cv2.putText(frame, text, label_pos, FONT,  
                   iFONT_SCALE, color, THICKNESS, cv2.LINE_AA)

    def __eq__(self, other):
        return is_overlapping(self.box, other.box)


def main(model_path="./best.pt", video_path="./assets/multi_objs.mp4", output_path="./results/multi_objs.mp4", standard_confidence=0.05):   

    shark_frame_tracker = []
    objects_frame_tracker = []
    
    # 1. Set up a model
    model = YOLO(model_path)
    model.to("cuda")

    # 2. Video Setup
    # Open the video file
    cap, video_writer = load_video(video_path, output_path)

    # Loop through the video frames
    frame_cnt = 1
    ids = {}
    while cap.isOpened():

        # Read a frame from the video
        success, frame = cap.read()
        print("success:", success)
        shark_frame_tracker.append(None)
        objects_frame_tracker.append([])
        prev_objs = {}
        if success:
            
            # Run YOLOv8 tracking on the frame, persisting tracks between frames
            results = model(frame)
            # print("THESE ARE THE RESULSTS:", results)
            # 1. Iterate the YOLO results
            for idx, r in enumerate(results):
                # Returns torch.Tensor(https://pytorch.org/docs/stable/tensors.html)
                # xywh returns the boxes in xywh format.
                # cpu() moves the object into cpu
                boxes = r.boxes.xywh.cpu()
                
                # Contains box plot information
                yolo_json_format = json.loads(r.tojson())
                # print("THIS IS THE JSON FORMAT", yolo_json_format)

                # 1-1. Construct all object list and shark list
                for obj in yolo_json_format:
                    print("\nTHIS IS AN OBJ:", obj)     
                    name = obj["name"]
                    cls = obj["class"]
                    box = obj["box"]
                    confidence = obj["confidence"] 
                    # obj_id = 1

                    
                    # algorithm to set ids
                    isOverlap = False
                    noPrevObj = True
                    print("frame_cnt", frame_cnt)
                    print(shark_frame_tracker)
                    print(objects_frame_tracker)
                    if len(shark_frame_tracker) == 1 and len(objects_frame_tracker) == 1: # if 1st object
                        print("this is the first frame")
                        obj_id = 1
                    else:
                        if name == "shark":    
                            prev_obj = shark_frame_tracker[frame_cnt-2] 
                        else:
                            # noSameClass = True
                            curr_frame = objects_frame_tracker[-1]
                            if len(curr_frame) != 0:
                                # there is previous obj
                                for past_obj in curr_frame[::-1]:
                                    print("past obj", past_obj)      
                                    if past_obj.name == name:
                                        prev_obj = objects_frame_tracker[frame_cnt - 2]
                                        noPrevObj = False
                        if noPrevObj and is_overlapping(obj["box"], prev_obj.box):
                            obj_id = prev_obj.id
                            isOverlap = True          
                     #  generates a new id if the object doesn't overlap (new object) or has a different class
                    if not isOverlap:
                        obj_id = len(ids) + 1
                        ids[obj_id] = True
                    
                    # Create a new General Object
                    new_obj = GeneralObject(name, cls, box, confidence, frame_cnt, obj_id)
                    print("for", name, "new_obj ID is:", obj_id)
                
                    # Append the object if it has a high possiblity of being a shark
                    if new_obj.name == 'shark' and new_obj.confidence > standard_confidence:
                        curr_shark = shark_frame_tracker[frame_cnt-1]

                        # Replace curr_shark to new_obj if new_obj's confidence is higher
                        if curr_shark is not None and curr_shark.confidence < new_obj.confidence:
                            shark_frame_tracker[frame_cnt-1] = new_obj
                        elif curr_shark is None:
                            shark_frame_tracker[frame_cnt-1] = new_obj
                    else:
                        objects_frame_tracker[frame_cnt-1].append(new_obj)
                    print("shark frame", shark_frame_tracker)
                    print("obj frame", objects_frame_tracker, "\n")

                    '''
                    print("shark frame", shark_frame_tracker)
                    print("obj frame", objects_frame_tracker, "\n")
                    prev_obj = None
                    if new_obj.name == "shark":          
                        prev_objs = shark_frame_tracker
                    else:
                        prev_objs = objects_frame_tracker
                    
                    print("currently on frame...", frame_cnt)
                    if len(prev_objs) == 1:
                        new_obj.id = id_count
                    else:
                        prev_obj = prev_objs[frame_cnt - 2]
                        curr_obj = prev_objs[frame_cnt - 1]
                        print("prev", prev_obj, "and", curr_obj)
                        if is_overlapping(prev_obj.box, curr_obj.box):
                                new_obj.id = prev_obj.id
                        else:
                                id_count += 1
                                new_obj.id = id_count
                    new_obj.update_ids()
            '''
            # print("all the shark frames", shark_frame_tracker)
            # print("all the object frames", objects_frame_tracker)
            # 2. Draw the tracking history for each loop
            prev_shark = None
            recently_detected_shark = None
            prev_person = None
            recently_detected_person = None
            
            # 2-1. Drawing the shark tracking line
            for i in range(len(shark_frame_tracker)):
                curr_shark = shark_frame_tracker[i]
                curr_objects = objects_frame_tracker[i]

                # 2-1-1. Draw the line
                if curr_shark is None:  # If no shark in the current frame, skip the loop
                    prev_shark = curr_shark
                    continue
                if prev_shark: # If the previous frame passes the bounding box detection test, draw the line
                    if prev_shark == curr_shark:
                        prev_shark.draw_line(frame, curr_shark, thickness=3)

                
                # Need interpolation here
                elif recently_detected_shark:
                    # If there is a recently detected shark and it passes the bounding box detection test, 
                    if recently_detected_shark == curr_shark: # Linear Interpolation
                        recently_detected_shark.draw_line(frame, curr_shark, color=INTERPOLATION_COLOR)
                    # If there is a recently detected shark and it does not pass the bounding box detection test, 
                    else:
                        pass
                        # recently_detected_shark.draw_line(frame, curr_shark, (255,255,153))

                # 2-1-2. Update recently_detected_shark
                if curr_shark:
                    recently_detected_shark = curr_shark
                
                # 2-1-3. Update prev_shark (this will be updated regardless of curr_shark exists or not)
                prev_shark = curr_shark

                for obj in curr_objects:
                    if obj.name == 'person':
                        if prev_person is not None and prev_person == obj:
                            prev_person.draw_line(frame, obj, thickness=3)
                        elif recently_detected_person is not None and recently_detected_person == obj:
                            recently_detected_person.draw_line(frame, obj, color=INTERPOLATION_COLOR)
                        recently_detected_person = obj
                    prev_person = obj

                

            # 2-2. Mark the currently detected objects
            for curr_obj in curr_objects:
                curr_obj.draw_box(frame)


            # 2-3. Mark the currently detected shark
            if curr_shark:
                curr_shark.draw_box(frame, (71,214,39))


            # 3. Write into the video file & increase the frame counter
            video_writer.write(frame)
            frame_cnt+=1

        else:
            # Break the loop if the end of the video is reached
            break
    
    # 4. Export the information for the metrices
    json_data = []
    for i in range(len(shark_frame_tracker)):
        json_data.append({"frame_cnt": i+1})
        
        curr_shark = shark_frame_tracker[i]
        curr_objects = objects_frame_tracker[i]
        
        if curr_shark:
            json_data[i]["shark"] = curr_shark.as_dict()
        else:
            json_data[i]["shark"] = None

        json_curr_objects = []
        for obj in curr_objects:
            json_curr_objects.append(obj.as_dict())
        
        json_data[i]["objects"] = json_curr_objects

        # 1. Calculate angles, distance, speed between all objects and the shark
        if len(curr_objects) == 0:
            continue
        else:
            pass
    
    json_objects = json.dumps(json_data, indent=4)

    with open("megan_multi.json", "w") as outfile:
        outfile.write(json_objects)

    # Release the video capture object and close the display window
    video_writer.release()
    cap.release()

if __name__ == "__main__":

    print("code is running...")
    main()
    print("code is done running...")
