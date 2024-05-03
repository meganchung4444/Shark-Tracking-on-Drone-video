import sys
import cv2
import numpy as np
import json
from ultralytics import YOLO
import supervision as sv
# Source: Sadli, R. (2023). Beginner’s Guide to Multi-Object Tracking with Kalman Filter.
from kf.tracker import Tracker
import seaborn as sns
# Written by Sunbeom (Ben) Kweon 
from general_object import GeneralObject


"""
Author: Megan Chung (MC)
Description: This program can handle videos with multiple objects. 
However, this program has not been fully tested with other videos so it may not produce the same results. 
"""

THICKNESS = 2
iFONT_SCALE = 0.7
FONT = cv2.FONT_HERSHEY_SIMPLEX
COLOR = (10, 20, 30)
INTERPOLATION_COLOR = (38, 146, 240)
LABEL_COLOR = (56, 56, 255)
# Source: Sadli, R. (2023). Beginner’s Guide to Multi-Object Tracking with Kalman Filter.
color_list = sns.color_palette('bright', 20) 
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


def main(model_path="best.pt", video_path="./assets/og-multi-objs.mp4", output_path="./results/output.mp4", standard_confidence=0.05):   

    shark_frame_tracker = []
    objects_frame_tracker = []
    
    # 1. Set up a model
    model = YOLO(model_path)
    model.to("cuda")

    # 2. Video Setup
    # Open the video file
    cap, video_writer = load_video(video_path, output_path)

    # Source: Sadli, R. (2023). Beginner’s Guide to Multi-Object Tracking with Kalman Filter.
    # MC - updated values when testing
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
    # MC - initialize Tracker object with parameters
    tracker = Tracker(tracker_params)
    # MC - initialize dictionary to keep track of objects' trajectories
    trajectory_dict = {}
    frame_cnt = 1

    while cap.isOpened():

        # Read a frame from the video
        success, frame = cap.read()
        shark_frame_tracker.append(None)
        objects_frame_tracker.append([])
        
        if success:
            
            # Run YOLOv8 tracking on the frame, persisting tracks between frames
            results = model(frame)

            # MC - Initialize list to store all the objects' center for a frame
            centers = []            
            
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

                    # MC - add the new detected object's center to list
                    centers.append(new_obj.center)                    

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
                
                # MC - Convert to numpy array and reshape to have 2 dimnesions: (automatic size), 2, 1
                centers = np.array(centers)
                centers = centers.reshape(-1, 2, 1)
                
                # MC - if objects detected (must have objects to have centers), manage track
                # manage_tracks: predict, perform track association, and update the state & handle unpaired tracks and detections
                if len(centers) > 0:
                    tracker.manage_tracks(centers)
                
                # MC - Loop through each track and update trajectory
                for track in tracker.tracks:    
                    if len(track.trace) > 0 and track.num_lost_dets <= 1:          
                        t_id = track.track_id
                        # MC - Get the existing trajectory if ID exists 
                        # or make an empty list for new ID
                        trajectory = trajectory_dict.get(t_id, []) 
                        # MC - Get the most recent position of the track
                        pos = track.updated
                        x, y = int(pos[0][0]), int(pos[1][0])
                        # MC - Append the position to the trajectory list and dictionary
                        trajectory.append((x, y))  
                        trajectory_dict[t_id] = trajectory
                        
                        # MC - Draw rectangle around the object
                        cv2.rectangle(frame, (x - 100, y - 100), (x + 100, y + 100), \
                                                    color_list[t_id%len(color_list)],5)
                        # MC - Display the track ID next to rectangle 
                        cv2.putText(frame, str(track.track_id), (x - 10, y - 20), 0, 5, \
                                                    color_list[t_id%len(color_list)], 5) 
                
                        # MC - Iterate each point of track's trace
                        # for k in range(len(track.trace)):  
                        #     # MC - Get x and y coordinate of current point                      
                        #     x1 = int(track.trace[k][0][0])
                        #     y1 = int(track.trace[k][1][0])

                        #     if k > 0:
                        #         # MC - Get x and y coordinate of previous point (if not the 1st point in trace)
                        #         x2 = int(track.trace[k - 1][0][0])
                        #         y2 = int(track.trace[k - 1][1][0])
                        #         # MC - Draw line connecting the current point
                        #         cv2.line(frame, (x1, y1), (x2, y2), color_list[t_id % len(color_list)], 10)
            
            # MC - Draw line for each tracjectory for each track
            for t_id, trajectory in trajectory_dict.items():
                if len(trajectory) > 1:
                    for i in range(1, len(trajectory)):
                        cv2.line(frame, trajectory[i - 1], trajectory[i], color_list[t_id % len(color_list)], thickness=3)
            
            # MC - Resize and display frame
            resize = ResizeWithAspectRatio(frame, width=1200, height=800)
            frame_nb_text= f"Frame:{frame_cnt}"                        
            cv2.putText(resize,frame_nb_text , (20, 40), \
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 1)
            
            cv2.imshow("Shark Tracking", resize) 
            cv2.resizeWindow("YOLOv8 Tracking", 1200, 800)

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

    main()