import numpy as np
from kf.kalman_filter import Kalman2d_mot
from kf.tracker_utils import track_association
    
class Track(object):
    def __init__(self, params, detection, trackId):    
        self.track_id = trackId        
        self.KF = Kalman2d_mot(params, detection)                
        self.num_lost_dets = 0
        self.trace = []

    def update(self, detection):
        self.updated = self.KF.update(detection)
    
    def predict(self):
        self.predicted = self.KF.predict()
       
class Tracker(object):
    def __init__(self, params):        
        self.params = params
        self.min_dist = self.params["min_dist"]
        self.max_lost_dets = self.params["max_lost_dets"]
        self.trace_length = self.params["trace_length"]
        self.id = self.params["id"]
        self.tracks = []       

    def manage_tracks(self, detections):                 
                
        #predict the tracks
        for i in range(len(self.tracks)):
            self.tracks[i].predict()

        pairs, unpaired_dets, unpaired_tracks =  track_association(self.tracks, \
                                                  detections, self.min_dist)
        
        #pairs indices: update the state
        for i,j in pairs:                         
            self.tracks[i].num_lost_dets = 0                                 
            self.tracks[i].update(detections[j])                                    
            
            if len(self.tracks[i].trace) >= self.trace_length:                
                self.tracks[i].trace = self.tracks[i].trace[:-1]
            self.tracks[i].trace.insert(0,self.tracks[i].updated)

        # unpaired tracks:
        #check and delete each track that has undergone lost detection for more than 
        #the value specified by the "self.max_lost_dets" parameter
        del_track = 0
        for i in unpaired_tracks:    
            if self.tracks[i - del_track].num_lost_dets > self.max_lost_dets:                             
                del self.tracks[i - del_track] 
                del_track += 1
            else:
                self.tracks[i - del_track].num_lost_dets += 1     
            
        #unpaired detections: create and initialize new tracks for unpaired detections
        for j in unpaired_dets:
            self.tracks.append(Track(self.params, detections[j], self.id))
            self.id += 1