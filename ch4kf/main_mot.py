import cv2
from tracker.tracker import Tracker
import detector
import  time        
import seaborn as sns
import json
import argparse
import numpy as np

color_list = sns.color_palette('bright', 10)
color_list = [(int(r*255), int(g*255), int(b*255)) for (r, g, b) in color_list]


def load_json(path):
    with open(path, "r") as f:
        config = json.load(f)
    return config

def main(args):

    VideoCap = cv2.VideoCapture(args.video_path)
    

    frame_nb=0
    while(True):
        # Read video frame
        ret, frame = VideoCap.read()
        if ret:                        
            centers = detector.detect(frame)
            
            #reshaped centers to have each element is a column vector of shape (2, 1)
            centers = centers[:, :, np.newaxis]

            tracker.manage_tracks(centers)

            for track in tracker.tracks:            
                if len(track.trace) > 0 and track.num_lost_dets <= 1:          
                    t_id = track.track_id
                    pos = track.updated

                    x, y = int(pos[0][0]),int(pos[1][0])                    
                                    
                    cv2.rectangle(frame, (x - 10, y - 10), (x + 10, y + 10), \
                                                color_list[t_id%len(color_list)],1)
                    
                    cv2.putText(frame, str(track.track_id), (x - 10, y - 20), 0, 0.5, \
                                                color_list[t_id%len(color_list)], 2)

                    for k in range(len(track.trace)):                        
                        x = int(track.trace[k][0][0])
                        y = int(track.trace[k][1][0])

                        cv2.circle(frame, (x, y), 3, \
                                   color_list[t_id%len(color_list)], - 1)
            
            frame_nb_text= f"Frame:{frame_nb}"                        
            cv2.putText(frame,frame_nb_text , (20, 40), \
                        cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 1)                    

            cv2.putText(frame, "www.machinelearningspace.com", (220, 300), \
                        cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 1)

            # Display the resulting tracking frame
            cv2.imshow('Tracking', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_nb+=1
        else:
            break
    
    VideoCap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Multi-Circle Tracking System')
    parser.add_argument('--tracker-params', type=str, \
                        default='./tracker/tracker_params.json')
    parser.add_argument('--video-path', type=str, \
                        default='./data/input_vid.mp4')    
    args = parser.parse_args() 

    # call the main function to execute
    main(args)