from .utils import draw_ellipse, draw_triangle # utils file inside this dir

class PlayerTracksDrawer:
    
    def __init__(self, team1_color = [255,245,238], team2_color = [128,0,0,]):
        
        self.default_player_team_id = 1
        
        self.team1_color = team1_color
        self.team2_color = team2_color
    
    def draw(self, video_frames, tracks, player_assignment, ball_acquisition):
        
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            
            # Return a dictionary containing for every frame 
            # the track_id(which is also the player_id) 
            # and the bbox of the tracked players
            player_dict = tracks[frame_num]
            
            player_assignment_for_frame = player_assignment[frame_num]
            
            player_id_has_ball = ball_acquisition[frame_num]
            
            # Draw players tracks
            for track_id, player_bbox in player_dict.items():
                team_id = player_assignment_for_frame.get(track_id, self.default_player_team_id)
                
                if team_id == 1:
                    color = self.team1_color
                else: 
                    color = self.team2_color
                
                if track_id == player_id_has_ball:
                    frame = draw_triangle(frame, player_bbox["bbox"], (0,0,255))
                
                frame = draw_ellipse(frame, player_bbox["bbox"], color, track_id)
              
            # Put the append outside the loop since we want to avoid adding N frames of the same thing
            # e.g. with 8 players detected in a particular moment -> 8 frames appended -> slow output video 
            output_video_frames.append(frame)
            
        return output_video_frames
                
            