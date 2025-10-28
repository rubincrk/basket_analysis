from ultralytics import YOLO
import sys
import supervision as sv
import numpy as np
import pandas as pd


sys.path.append("../")
from utils import read_stub, save_stub

class BallTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
    
    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i+batch_size]
            batch_detections = self.model.predict(batch_frames, conf=0.5)
            detections+=batch_detections
            
        return detections
    
    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        
        tracks = read_stub(read_from_stub, stub_path)
        if tracks is not None:
            if len(tracks) == len(frames):
                return tracks
            
        detections = self.detect_frames(frames)
        tracks = []
        
        for frame_num, detection in enumerate(detections):
            cls_name = detection.names
            cls_name_inv = {v:k for k,v in cls_name.items()}
            
            detection_supervision = sv.Detections.from_ultralytics(detection)
            tracks.append({})
            chosen_bbox = None
            max_confidence = 0
            
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                confidence = frame_detection[2]
                
                if cls_id == cls_name_inv["Ball"]:
                    if max_confidence < confidence:
                        chosen_bbox = bbox
                        max_confidence = confidence
            
            if chosen_bbox is not None:
                # The '1' is hardcoded as there is 1 object(track_id) we care about -> the ball
                tracks[frame_num][1] = {"bbox":chosen_bbox}
        
        save_stub(stub_path, tracks)
        return tracks
    
    def remove_wrong_detections(self, ball_positions):
        
        maximum_allowed_distance = 25 # Pixels per frame
        # So if the ball det disappears for 3 frame we will have 25*3
        last_good_frame_index = -1
        
        for i in range(len(ball_positions)):
            # We use get (1,) because we have track_id = 1, returns empty dict otherwise
            # And we take the bbox of the ball tracked
            current_bbox = ball_positions[i].get(1,{}).get("bbox", [])
            
            if len(current_bbox) == 0:
                continue
            
            # Handling the 1st detection of the ball
            if last_good_frame_index == -1:
                last_good_frame_index=i
                continue
            
            last_good_box = ball_positions[last_good_frame_index].get(1,{}).get("bbox", [])
            frame_gap = i - last_good_frame_index
            adjusted_max_distance = maximum_allowed_distance * frame_gap
            
            # Calculate the dist between the last good bbox and the current position
            # using their (x,y), that's why we take the first 2 [:2] elements
            if np.linalg.norm(np.array(last_good_box[:2]) - np.array(current_bbox[:2])) > adjusted_max_distance:
                ball_positions[i] = {}
            else:
                last_good_frame_index = i
            
        return ball_positions
            
    
    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [ x.get(1,{}).get("bbox", []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=["x1", "y1", "x2", "y2"])
        
        # Interpolate missing values
        '''
        It fills the NaN values by linearly interpolating between the previous and next valid numbers.
        '''
        df_ball_positions = df_ball_positions.interpolate()
        
        '''
        bfill() means backward fill:
        If the first few frames were missing (no previous data), it fills them using the first known value.
        '''
        df_ball_positions = df_ball_positions.bfill()
        
        # We rebuild the same structure we started with
        '''
        We have a pandas dataframe that looks like this:
        | frame | x1    | y1    | x2    | y2    |
        | 0     | 100.0 | 150.0 | 130.0 | 180.0 |
        | 1     | 105.0 | 152.0 | 135.0 | 182.0 |
        | 2     | 110.0 | 156.0 | 140.0 | 186.0 |
        | 3     | 115.0 | 160.0 | 145.0 | 190.0 |
        
        The goal is to go back to:
        [
            {1: {"bbox": [100.0,150.0,130.0,180.0]}},
            {1: {"bbox": [105.0,152.0,135.0,182.0]}},
            ...
        ]
        
        df_ball_positions.to_numpy() converts the DataFrame values into a NumPy 2D array with shape (num_frames, 4):

        array([
            [100.,150.,130.,180.],
            [105.,152.,135.,182.],
            [110.,156.,140.,186.],
            [115.,160.,145.,190.]
        ])
        - Which gives a compact numerical matrix (no index, no column labels).
        - Easy to loop over rows or apply vectorized operations.
        - Faster and memory-lighter than iterating over Pandas rows.
        
        tolist() simply converts that NumPy array into a plain list of Python lists:

        [
            [100.0,150.0,130.0,180.0],
            [105.0,152.0,135.0,182.0],
            [110.0,156.0,140.0,186.0],
            [115.0,160.0,145.0,190.0]
        ]
        
        Now you can easily rebuild the structure with a simple list comprehension:

        [{1: {"bbox": x}} for x in df_ball_positions.to_numpy().tolist()]

        Each x here is one [x1, y1, x2, y2] list.

        '''
        ball_positions = [ {1:{"bbox" : x}} for x in df_ball_positions.to_numpy().tolist()]
        
        return ball_positions
        