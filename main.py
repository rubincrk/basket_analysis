from utils import read_video, save_video
from trackers import PlayerTracker, BallTracker
from drawers import PlayerTracksDrawer, BallTracksDrawer, TeamBallControlDrawer
from team_assigner import TeamAssigner
from ball_acquisition import BallAquisitionDetector



def main():
    
    #Read video
    video_frames = read_video("input_videos/video_1.mp4")
    
    #Initialize Tracker
    player_tracker = PlayerTracker("models/player_detector.pt")
    ball_tracker = BallTracker("models/ball_detector_model.pt")
    
    #Run Trackers
    player_tracks = player_tracker.get_object_tracks(
        video_frames,
        read_from_stub=True,
        stub_path="stubs/player_track_stubs.pkl"
    )
    
    #Ball Tracker
    ball_tracks = ball_tracker.get_object_tracks(
        video_frames,
        read_from_stub=True,
        stub_path="stubs/ball_track_stubs.pkl"
    )
    
    # Remove wrong ball detections
    ball_tracks = ball_tracker.remove_wrong_detections(ball_tracks)
    
    # Interpolate ball tracks
    ball_tracks = ball_tracker.interpolate_ball_positions(ball_tracks)
    
    # Assign player teams
    team_assigner = TeamAssigner()
    player_assignment = team_assigner.get_player_teams_across_frames(
        video_frames, 
        player_tracks, 
        read_from_stub = True,
        stub_path = "stubs/player_assignment_stub.pkl"
        )
    
    # Ball acquisition
    ball_acquisition_detector = BallAquisitionDetector()
    ball_acquisition = ball_acquisition_detector.detect_ball_possession(player_tracks, ball_tracks)
    #print(ball_acquisition)
    
    # Draw output
    # Initialize Drawers
    player_tracks_drawer = PlayerTracksDrawer()
    ball_tracks_drawer = BallTracksDrawer()
    team_ball_control_drawer= TeamBallControlDrawer()
    
    # Draw Object Tracks
    output_video_frames = player_tracks_drawer.draw(
        video_frames, 
        player_tracks, 
        player_assignment,
        ball_acquisition
        )
    output_video_frames = ball_tracks_drawer.draw(output_video_frames, ball_tracks)
    
    # Draw team ball control
    #output_video_frames = team_ball_control_drawer.draw(output_video_frames, player_assignment, ball_acquisition)
    
    #Save video
    save_video(output_video_frames,"output_videos/output_video.avi")
    


if __name__ == "__main__":
    main()