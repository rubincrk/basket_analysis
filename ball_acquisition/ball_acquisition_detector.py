import sys
sys.path.append("../")
from utils import measure_distance, get_center_of_bbox


class BallAquisitionDetector:
    def __init__(self):
        '''
        For each frame and player, compute:

            !) Distance from the ball center to several key points on/around the player bbox.

            2) Containment ratio: area(ball ∩ player) / area(ball).

        If the distance is small and containment ratio is high for ≥ min_frames consecutively → the player is deemed to have acquired the ball.
        
        '''
        
        self.possession_threshold = 50 # pixels: ball must be within this distance of player box keypoints
        self.min_frames = 11 # frames: must satisfy rules this many frames in a row
        self.containment_threshold = 0.8 # fraction: % of ball area inside player box
        
        
    '''
    Create a small set of representative points on the player box (corners, edge midpoints, and optionally points aligned with the ball's x/y through the box). 
    We'll measure the ball's distance to the closest of these, which is a fast approximation of “how close to the player” the ball is.

    Example

    player_bbox = (100, 50, 180, 190) (x1, y1, x2, y2)

    ball_center = (170, 120)

    Since y1 < 120 < y2, add (x1, 120)=(100,120) and (x2,120)=(180,120).
    Since x1 < 170 < x2, add (170, y1)=(170,50) and (170,y2)=(170,190).
    Then corners and midpoints are added.
    
    FIRST CONDITION:
        What we check:

        “Is the ball vertically within the player's height range?”

        i.e.,
        does the ball's y-coordinate lie between the top and bottom of the player box?

        If so, it means the ball is roughly beside the player (same height zone).

        It adds Two points:

        (x1, ball_center_y) → point on the left edge of the box, exactly at the ball's vertical level.

        (x2, ball_center_y) → point on the right edge of the box, same vertical level.

        So you get two points horizontally aligned with the ball's center.
        
        (y increases ↓)
                  ball_center ● (170,120)
                              |
        (x1,y1) +-------------+--------------+ (x2,y1)
                |                            |
                |         player box         |
                |                            |
        (x1,y2) +----------------------------+ (x2,y2)
 
        Because the ball's y=120 is between y1 and y2,
        we add those two points:

        (x1,120)  ●---------------------------●  (x2,120)


        Now we can measure distance from the ball center to the closest side.

        If the ball is right next to the player's left or right boundary,
        the shortest distance is a simple horizontal gap.

    SECOND CONDITION:
        “Is the ball horizontally within the player's width range?”

        If the ball's x lies between the left and right of the box,
        the ball is vertically aligned (above or below the player).

        It adds Two points:

        (ball_center_x, y1) → point on the top edge of the box, same x as the ball.

        (ball_center_x, y2) → point on the bottom edge of the box, same x as the ball.
        
                    (ball_center_x=130)
                          |
            +-------------+-------------+
            |             |             |
     (x1,y1)+-------------+-------------+
            |             |             |
            |   player    |   box       |
            |             |             |
     (x1,y2)+-------------+-------------+
                          ●  ball_center (130,220)

        Since x=130 is within the player's left/right range,
        we add the points directly above and below the ball on the player's top and bottom edges.
    
    These extra points ensure that for any position of the ball relative to the player box:

    If the ball is beside the player → shortest distance is measured horizontally.

    If the ball is above/below the player → shortest distance is measured vertically.

    If the ball is diagonal → corners and midpoints already cover that.
    
    '''    
    
    def get_key_basketball_player_assignment_points(self, player_bbox, ball_center):
        ball_center_x = ball_center[0]
        ball_center_y = ball_center[1]
        
        x1, y1, x2, y2 = player_bbox
        width = x2-x1
        height = y2-y1
        
        output_points = []
        
        '''
        This first condition means:

        “If the ball is at the same height as the player (not above or below),
        then add two points — one on the left side, one on the right side — at the same height as the ball.”

        So we can measure how far the ball is from the left or right side of the player.
        ● ball
        |
        |
        +---------+
        | Player  |
        +---------+

        '''
        if ball_center_y > y1 and ball_center_y < y2:
            output_points.append((x1, ball_center_y))
            output_points.append((x2, ball_center_y))
        
        '''
        This second condition means:

        “If the ball is in front of the player horizontally (not to the side),
        then add two points — one on the top, one on the bottom — at the same x as the ball.”

        So we can measure how far the ball is from the top or bottom of the player.
        ● ball
            |
            +---------+
            | Player  |
            +---------+

        '''
        if ball_center_x > x1 and ball_center_x < x2:
            output_points.append((ball_center_x, y1))
            output_points.append((ball_center_x, y2))
            
        output_points += [
            (x1,y1), # top left corner
            (x2,y1), # top right corner
            (x1,y2), # bottom left corner
            (x2,y2), # bottom right corner
            (x1+width//2,y1), # top center
            (x1+width//2,y2), # bottom center
            (x1,y1+height//2), # left center
            (x2,y1+height//2) # right center
        ]
        
        
        return output_points
    
    '''
    Finds the closest key point to the ball center — a cheap proxy for proximity between ball and player.

    measure_distance(a,b) is Euclidean:
    sqrt( (ax - bx)^2 + (ay - by)^2 ).

    Using the earlier example:

    ball_center = (170,120)

    One key point (180,120) is only 10 px away (closest):
    d = (170-180)^2 + (120-120)^2 = (-10)^2 + 0^2 = 10

    If possession_threshold = 50, then distance 10 < 50 → passes proximity test.
    
    The Euclidean distance is simply the length of the straight line between two points:
    It captures how spatially near two things are on the 2D frame.
    '''
    def find_minimun_distance_to_ball(self, ball_center, player_bbox):
        key_points = self.get_key_basketball_player_assignment_points(player_bbox, ball_center)
        
        return min(measure_distance(ball_center, key_point) for key_point in key_points)
        
        # Alternative
        '''
            min = 99999
            for key_point in key_points:
                distance = measure_distance(ball_center, key_point)
                if min > distance:
                    min = distance 
            return min
        '''
        
    
    '''
    Computes the fraction of the ball area that lies inside the player bbox. If this ratio ≥ containment_threshold (e.g., 0.8), we consider the ball mostly inside the player box.

    Example A — partial overlap (0.75)

    player_bbox = (100, 50, 180, 190)

    ball_bbox = (165,105,185,125) → width 20, height 20, area 400

    Intersection:

    ix1=max(100,165)=165

    iy1=max(50,105)=105

    ix2=min(180,185)=180

    iy2=min(190,125)=125

    iw=180-165=15, ih=125-105=20, inter_area=300

    containment_ratio = 300/400 = 0.75 (< 0.8 → fail containment test)

    Example B — fully inside (1.0)

    ball_bbox = (166,106,176,116) → still inside player box.

    Intersection == ball → ratio 1.0 (≥ 0.8 → pass)
    '''
    def calculate_ball_containment_ratio(self, player_bbox, ball_bbox):
        
       px1,py1,px2,py2 = player_bbox
       bx1,by1,bx2,by2 = ball_bbox
       
       # Width * height
       ball_area = (bx2-bx1)*(by2-by1)
       
       #player_area = (px2-px1)*(py2-py1)
       
       intersection_x1 = max(px1,bx1)
       intersection_y1 = max(py1,by1)
       intersection_x2 = min(px2,bx2)
       intersection_y2 = min(py2,by2)
       
       if intersection_x2 < intersection_x1 or intersection_y2 < intersection_y1:
           return 0
       
       intersection_area = (intersection_x2-intersection_x1) * (intersection_y2-intersection_y1)
       
       containment_ratio = intersection_area/ball_area
       
       return containment_ratio
       
    
    def find_best_candidate_for_position(self, ball_center, player_tracks_frame, ball_bbox):
        
        high_containment_players = []
        regular_distance_players = []
        
        for player_id, player_info in player_tracks_frame.items():
            player_bbox = player_info.get("bbox", [])
            
            if not player_bbox:
                continue
            
            containment = self.calculate_ball_containment_ratio(player_bbox, ball_bbox)
            min_distance = self.find_minimun_distance_to_ball(ball_center, player_bbox)
            
            if containment > self.containment_threshold:
                high_containment_players.append((player_id, containment))
            else:
                regular_distance_players.append((player_id, min_distance))
                
        # First priority: High containment players
        # If any exist, pick the one with the highest containment
        if high_containment_players:
            best_candidate = max(high_containment_players, key=lambda x:x[1])
            return best_candidate[0] # return player_id and not the containment_id
        
        # Second priority: regular distance
        # If nobody passes containment, look at everyone else’s min distance.
        # Pick the smallest distance, and return that player only if distance < possession_threshold.
        if regular_distance_players:
            best_candidate = min(regular_distance_players, key= lambda x:x[1])
            if best_candidate[1] < self.possession_threshold:
                return best_candidate[0] # Return player_id instead of min_distance
        
        return -1
    
    def detect_ball_possession(self, player_tracks, ball_tracks):
        num_frames = len(ball_tracks)
        # No one has the ball
        possesion_list = [-1] * num_frames
        consecutive_possession_count = {}
        
        for frame_num in range(num_frames):
            ball_info = ball_tracks[frame_num].get(1,{}) # 1 is track_id of the ball
            
            if not ball_info:
                continue
            
            ball_bbox = ball_info.get("bbox", [])
            if not ball_bbox:
                continue
            
            ball_center = get_center_of_bbox(ball_bbox)
            
            best_player_id = self.find_best_candidate_for_position(
                ball_center, 
                player_tracks[frame_num], 
                ball_bbox
            )
            
            if best_player_id != -1:
                number_of_consecutive_frames = consecutive_possession_count.get(best_player_id,0)+1
                consecutive_possession_count = {best_player_id:number_of_consecutive_frames}
                
                if consecutive_possession_count[best_player_id] >= self.min_frames:
                    possesion_list[frame_num] = best_player_id
            else:
                consecutive_possession_count = {}
        
        return possesion_list
                