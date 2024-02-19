import cv2
from detector import detect
from kalman_filter import KalmanFilter2d
import matplotlib.pyplot as plt
import numpy as np
import json
import argparse
import os
from sklearn.metrics import mean_squared_error

def load_json(path):
    with open(path, "r") as f:
        config = json.load(f)
    return config

def main(args):

    # Load parameters from JSON file
    kalman_params = load_json(args.tracker_params)         
     
    # Load true object positions
    GT= np.load(args.gt_path)

    meas=[]
    pred=[]
    est=[]
    
    # Set to 0 to avoiod adding additional noise
    noise_std = 5

    track = False
        
    try:
        # Create opencv video capture object    
        VideoCap = cv2.VideoCapture(args.video_path)
        
        if not VideoCap.isOpened():
            raise Exception("Error: Could not open the video file.")
        
        fr=0
        while(True):
            # Read frame
            ret, frame = VideoCap.read()
            
            if not ret:
                break
            
            # Detect the circle using detect() function from the detector module
            center = detect(frame)[0].reshape((2,1)) 

            if not track:
                KF = KalmanFilter2d(kalman_params,center)
                track = True

            # If we detect the cicle, track it
            if (len(center) > 0):

                # Get the measurement positions 
                x_meas = center[0] + np.random.normal(0, noise_std)
                y_meas = center[1] + np.random.normal(0, noise_std)             
                meas.append((x_meas,y_meas))
                                            
                # Obtain the groundtruth for the x and y coordinates of the tracked circle 
                gt = GT[fr]
                x_tp = gt[0]
                y_tp = gt[1]      
                
                #### Predict next position based on the newly updated position                
                predicted_state = KF.predict()
                x_pred, y_pred = predicted_state[0], predicted_state[1]
                pred.append((x_pred, y_pred))

                #### Update the state position based on the prediction and new measurement                              
                updated_state = KF.update(np.array([x_meas,y_meas]))    
                x_est, y_est =  updated_state[0], updated_state[1]                         
                est.append((x_est,y_est))         
                                                                                                
                #### Draw a rectangle for the measured position
                cv2.rectangle(frame, (int(x_meas[0] - 15), int(y_meas[0] - 15)), \
                                        (int(x_meas[0] + 15), int(y_meas[0] + 15)), (0, 0, 255), 2)

                #### Draw a circle for the ground truth position
                cv2.circle(frame, (int(x_tp), int(y_tp)), 10, (0, 255, 255), 2)

                #### Draw a rectangle for the estimated position
                cv2.rectangle(frame, (int(x_est[0] - 15), int(y_est[0] - 15)), \
                                        (int(x_est[0] + 15), int(y_est[0] + 15)), (255, 0, 0), 2)

                #### Add legends
                cv2.rectangle(frame, (10, 30), (30,50), (0, 0, 255), 2)   
                cv2.putText(frame, "Measured Position", (40, 45), 0, 0.5, (0, 0, 0), 1)

                cv2.rectangle(frame, (10, 60), (30,80), (255, 0, 0), 2) 
                cv2.putText(frame, "Estimated Position", (40, 75), 0, 0.5, (0, 0, 0), 1)

                cv2.circle(frame, (20, 100), 10, (0, 255, 255), 2) 
                cv2.putText(frame, "True Position", (40, 105), 0, 0.5, (0, 0, 0), 1)
                                
                cv2.putText(frame, "Frame: "+ str(fr), (450,20), 0, 0.5, (0,0,255), 2)

            cv2.imshow('2-D Object Tracking Kalman Filter', frame)               
            if cv2.waitKey(20) & 0xFF == ord('q'):                            
                break        
                
            fr +=1

    except Exception as error:
        print("An error occurred:", str(error))
        
    finally:
        # Release the video capture object and close all active window created by OpenCV
        if VideoCap is not None:                   
            VideoCap.release()
        cv2.destroyAllWindows() 
    
    print('The circle tracking has been performed successfully.')
        
    ###  Plot Tracking Performance in the x and y directions ###    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle('2-D Kalman Filter Tracking Performance \n (Frames vs Positions)', \
                 fontsize=15, weight='bold')

    ax1.invert_yaxis()
    ax2.invert_yaxis()
    
    t = np.arange(0, len(pred), 1)
    pred= np.array(pred)
    est= np.array(est)
    meas= np.array(meas)

    # Plot the first subplot    
    ax1.plot(t, GT[:, 0], label='True x Position', color='yellow', linestyle='-', linewidth=2)
    ax1.plot(t, meas[:, 0], label='Measured x position', color='r', linestyle='-', linewidth=1)    
    ax1.plot(t, est[:, 0], label='Estimated x position', color='b', linestyle='-', linewidth=1)
    
    ax1.set_title('Comparison in the x direction', fontsize=15)
    ax1.set_xlabel('Frames', fontsize=15)
    ax1.set_ylabel('x positions (pixels)', fontsize=15)
    ax1.legend(fontsize=15)
    ax1.legend(loc='lower left')
    
    # Plot the second subplot
    #ax2.plot(t, pred[:, 1], label='Predicted y position', color='r', linestyle='--', linewidth=1)
    ax2.plot(t, GT[:, 1], label='True y Position', color='yellow', linestyle='-', linewidth=2)
    ax2.plot(t, meas[:, 1], label='Measured y position', color='r', linestyle='-', linewidth=1)
    ax2.plot(t, est[:, 1], label='Estimated y position', color='b', linestyle='-', linewidth=1)
    
    ax2.set_title('Comparison in the y direction', fontsize=15)
    ax2.set_xlabel('Frames', fontsize=15)
    ax2.set_ylabel('y positions (pixels)', fontsize=15)
    ax2.legend(fontsize=15)
    ax2.legend(loc='lower left')
    
    #### Plot Comparison in 2D Space ###
    fig, ax3 = plt.subplots()    
    ax3.set_title('Comparison of Estimated (KF), Measured, and True Trajectory \
Positions in 2D Space',fontsize=12, weight='bold')
    ax3.invert_yaxis()    
    ax3.plot(GT[:, 0], GT[:, 1], label='True Position', color='yellow', \
                                                            linestyle='-', linewidth=2)
    ax3.plot(meas[:, 0], meas[:, 1], label='Measured position', color='r', marker='*', \
                                                    markersize=4, linestyle='None', linewidth=1)
    ax3.plot(est[:, 0], est[:, 1], label='Estimated position', color='b', marker='o', \
                                                    markersize=3, linestyle='None', linewidth=2) 
         
    # Add axis labels and title
    ax3.set_xlabel('x positions (pixels)', fontsize=15)
    ax3.set_ylabel('y positions (pixels)', fontsize=15)
    ax3.legend(fontsize=15)    
    ax3.legend(loc='lower left')

    #### Calculate mean squared error (MSE)
    mse_est_x = mean_squared_error(GT[:, 0],est[:, 0])    
    mse_est_y = mean_squared_error(GT[:, 1],est[:, 1])

    mse_meas_x = mean_squared_error(GT[:, 0],meas[:, 0])
    mse_meas_y = mean_squared_error(GT[:, 1],meas[:, 1])

    print(f"mse_meas_x: {mse_meas_x}")
    print(f"mse_est_x: {mse_est_x}")
    print(f"mse_meas_y: {mse_meas_y}")        
    print(f"mse_est_y: {mse_est_y}")        
    
    plt.show()

if __name__ == "__main__":
    # execute main
    parser = argparse.ArgumentParser(description='MOT System Using Yolov3')
    parser.add_argument('--tracker-params', type=str, \
                        default='tracker_params.json')
    parser.add_argument('--video-path', type=str, \
                        default='./resources/input_vid.mp4')
    parser.add_argument('--gt-path', type=str, \
                        default='./resources/input_vid_gt.npy')
    args = parser.parse_args()   

    main(args)