import sys
import cv2
import numpy as np
import json
from ultralytics import YOLO
import supervision as sv
from ch4kf.tracker import Tracker
# from ch5kf.tracker import Tracker
import seaborn as sns

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
color_list = sns.color_palette('bright', 20) # 10 before
color_list = [(int(r*255), int(g*255), int(b*255)) for (r, g, b) in color_list]

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

# def get_box_center(box):
#     x1 = box["x1"]
#     x2 = box["x2"]
#     y1 = box["y1"]
#     y2 = box["y2"]
#     return (int(x1 + ((x2 - x1)//2)), int(y1 + ((y2 - y1)//2)))
def get_box_center(box):
    x1 = box["x1"]
    x2 = box["x2"]
    y1 = box["y1"]
    y2 = box["y2"]
    return np.array([int(x1 + ((x2 - x1)//2)), int(y1 + ((y2 - y1)//2))])

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



def main(model_path="best.pt", video_path="./assets/multi_objs.mp4", output_path="./results/multi_objs.mp4", standard_confidence=0.05):   

    shark_frame_tracker = []
    objects_frame_tracker = []
    
    # 1. Set up a model
    model = YOLO(model_path)
    model.to("cuda")

    # 2. Video Setup
    # Open the video file
    cap, video_writer = load_video(video_path, output_path)


    trajectory_dict = {}
    # Loop through the video frames
    frame_cnt = 1
    tracker_params = {
    "u": 1, 
    "dt":0.1,    
    "std_acc": 0.1, 
    "std_meas_x": 0.002,
    "std_meas_y": 0.002,
    "min_dist": 0.5, 
    "min_iou": 0.01, 
    "max_lost_dets": 15, 
    "trace_length":40, 
    "id": 0
    }

    all_boxes = []
    tracker = Tracker(tracker_params)
    id_dict = {}
    while cap.isOpened():

        # Read a frame from the video
        success, frame = cap.read()
        shark_frame_tracker.append(None)
        objects_frame_tracker.append([])
        
        if success:
            
            # Run YOLOv8 tracking on the frame, persisting tracks between frames
            results = model(frame)

            centers = []
            iou_list = []
            

            # for testing
            # if frame_cnt == 100:
            #     break
            
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
                    centers.append(new_obj.center)
                    all_boxes.append(new_obj.box)  
                    

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
                centers = np.array(centers)
                centers = centers.reshape(-1, 2, 1)
                

                if len(centers) > 0:
                    tracker.manage_tracks(centers)
                
                # applying kalman filter 
                for track in tracker.tracks:    
                    if len(track.trace) > 0 and track.num_lost_dets <= 1:          
                        t_id = track.track_id
                        trajectory = trajectory_dict.get(t_id, [])  # Get the existing trajectory or initialize an empty list
                        # most recent position of the track
                        pos = track.updated
                        x, y = int(pos[0][0]), int(pos[1][0])
                        trajectory.append((x, y))  
                        trajectory_dict[t_id] = trajectory
                        
                        cv2.rectangle(frame, (x - 100, y - 100), (x + 100, y + 100), \
                                                    color_list[t_id%len(color_list)],5)
                        
                        cv2.putText(frame, str(track.track_id), (x - 10, y - 20), 0, 5, \
                                                    color_list[t_id%len(color_list)], 5) # 5 from 0.5
                        
                        # print("track.trace:", track.trace)
                        for k in range(len(track.trace)):                        
                            x = int(track.trace[k][0][0])
                            y = int(track.trace[k][1][0])

                            # cv2.circle(frame, (x, y), 5, \
                            #         color_list[t_id%len(color_list)], - 1) 
                            # udpated pixel size to 5 from 3
                            
                            if k > 0:
                                x2 = int(track.trace[k - 1][0][0])
                                y2 = int(track.trace[k - 1][1][0])
                                cv2.line(frame, (x, y), (x2, y2), color_list[t_id % len(color_list)], 10)
            
            for t_id, trajectory in trajectory_dict.items():
                if len(trajectory) > 1:
                    for i in range(1, len(trajectory)):
                        cv2.line(frame, trajectory[i - 1], trajectory[i], color_list[t_id % len(color_list)], thickness=5)
            
            #  2. Draw the tracking history for each loop
            prev_shark = None
            recently_detected_shark = None
            
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

            # 2-2. Mark the currently detected objects
            for curr_obj in curr_objects:
                curr_obj.draw_box(frame)


            # 2-3. Mark the currently detected shark
            if curr_shark:
                curr_shark.draw_box(frame, (71,214,39))

            resize = ResizeWithAspectRatio(frame, width=1200, height=800)
            print("FRAME COUNT:", frame_cnt)
            frame_nb_text= f"Frame:{frame_cnt}"                        
            cv2.putText(resize,frame_nb_text , (20, 40), \
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 1)
            
            cv2.imshow("Shark Tracking", resize) # not resize
            # cv2.resizeWindow("YOLOv8 Tracking", 1200, 800)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            # 3. Write into the video file & increase the frame counter
            video_writer.write(frame)
            frame_cnt+=1

        else:
            print("video has ended")
           
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

    with open("output.json", "w") as outfile:
        outfile.write(json_objects)
    

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