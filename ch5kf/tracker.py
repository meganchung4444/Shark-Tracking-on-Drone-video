import numpy as np
# from tracker.kfilter_bbox import KalmanBBoxMot
from ch4kf.kalman_filter import Kalman2d_mot
from ch4kf.tracker_utils import track_association
    
class Track(object):
    def __init__(self, params, detection, trackId):                
        self.track_id = trackId        
        # self.KF = KalmanBBoxMot(params, detection) 
        self.KF = Kalman2d_mot(params, detection)               
        self.num_lost_dets = 0
        self.trace = []        
         
    def predict(self):
        self.predicted = self.KF.predict()

    def update(self, detection):
        self.updated = self.KF.update(detection)      
        
class Tracker(object):
    def __init__(self, params):
        self.params = params
        self.min_iou = self.params["min_iou"]
        self.max_lost_dets = self.params["max_lost_dets"]
        self.trace_length = self.params["trace_length"]
        self.id = self.params["id"]
        self.tracks = []       

    def manage_tracks(self, detections):   
        
        # detections= dets_all['dets']

        for i in range(len(self.tracks)):
            self.tracks[i].predict() 
        
        pairs, unpaired_dets, unpaired_tracks = track_association(self.tracks, \
                                                        detections, self.min_iou)
        
        for i,j in pairs:                         
            self.tracks[i].num_lost_dets = 0
            self.tracks[i].update(detections[j])

            # updated_state=self.tracks[i].updated[:4]
            # updated_state=np.concatenate([updated_state, info[j].reshape((-1,1))])     
            
            if len(self.tracks[i].trace) >= self.trace_length:                
                self.tracks[i].trace = self.tracks[i].trace[:-1]
            self.tracks[i].trace.insert(0,updated_state)        
                                
        del_track = 0
        for i in unpaired_tracks:   
            if self.tracks[i - del_track].num_lost_dets > self.max_lost_dets:                             
                del self.tracks[i - del_track] 
                del_track += 1            
            else:
                self.tracks[i- del_track].num_lost_dets += 1     
                    
        # create and initialize new trackers for unpaired detections
        for i in unpaired_dets:        
            self.tracks.append(Track(self.params, detections[i], self.id))                     
            self.id += 1      