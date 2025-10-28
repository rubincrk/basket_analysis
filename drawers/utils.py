import cv2
import sys
import numpy as np
sys.path.append("../")
from utils.bbox_utils import get_bbox_width, get_center_of_bbox


# OpenCV coordinate system
'''

(0,0) -------------> x (columns -> increases to the right)
|
|
|
y (rows -> increases downward)

So:
    increasing x → move right
    decreasing x → move left
    increasing y → move down
    decreasing y → move up
    
| Operation | Result                     | Meaning                           |
| --------- | -------------------------- | --------------------------------- |
| `x + N`   | move **right** by N pixels | object shifts to the right        |
| `x - N`   | move **left** by N pixels  | object shifts to the left         |
| `y + N`   | move **down** by N pixels  | object shifts lower in the image  |
| `y - N`   | move **up** by N pixels    | object shifts higher in the image |

    
    
'''

def draw_triangle(frame, bbox, color):
    '''
    [x1, y1, x2, y2]:

    y1 (or bbox[1]) = top edge of the bbox (head/top of object).

    x_center = midpoint between x1 and x2.
    '''
    y = int(bbox[1])
    x, _ = get_center_of_bbox(bbox)
    
    '''
    (x, y) is the center of the top edge of the box.
    smaller y → higher up on the image
    '''
    
    triangle_points = np.array([
        [x,y],
        [x-10,y-20], # upper-left 
        [x+10, y-20] # upper-right
    ])
    cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED) #filling
    cv2.drawContours(frame, [triangle_points], 0, (0,0,0), 2) #border
    return frame


def draw_ellipse(frame, bbox, color, track_id=None):
    y2 = int(bbox[3])
    x_center, _ = get_center_of_bbox(bbox)
    width = get_bbox_width(bbox)
    
    # It takes ellipse(image, center, axes_size, angle, start_angle, end_angle, color, thickness)
    cv2.ellipse(frame, 
                center = (x_center, y2), # y2 represents where the feet of the player are since (x2 , y2) represents the bottom right of the bbox
                axes = (int(width),int(0.35*width)), # just following the ellipse shape, the y-axis is not as large as the x-axis, so it is vertically compressed
                angle = 0, # whether or not the ellipse is going to be stretched
                startAngle = -45, 
                endAngle = 235, 
                color = color, 
                thickness = 2,
                lineType = cv2.LINE_4
    )
    
    rectangle_width = 40
    rectangle_height = 20
    x1_rect = int(x_center-rectangle_width//2) # starts from the center and goes a bit on the left
    x2_rect = int(x_center+rectangle_width//2) # starts from the center and goes a bit on the right
    y1_rect = int(y2 - rectangle_height//2) + 15 # starts from the bottom-right of the bbox and goes a bit up then +15 moves the rectangle down (further below the feet). This is the top-left edge
    y2_rect = int(y2 + rectangle_height//2) + 15 # starts from the bottom-right of the bbox and goes a bit down then +15 moves the rectangle down (further below the feet). This is the bottom-right edge
    
    if track_id is not None:
        # Remeber that the rectangle is drawn using 2 opposite corner points: top-left and bottom-right
        '''
            (y ↑)
            |
            |  (x1,y1) ●───────┐
            |          │       │   ← height = y2 − y1
            |          └───────● (x2,y2)
            |
            └─────────────────────────► x

        '''
        cv2.rectangle(frame, (x1_rect, y1_rect), (x2_rect, y2_rect), color, cv2.FILLED)
        
        x1_text = x1_rect + 12
        if track_id > 99:
            x1_text -= 10 # So we have more space for the 3 digit track_id
            
        cv2.putText(frame, 
                    str(track_id), 
                    (int(x1_text), int(y1_rect + 15)), 
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0,165,255),
                    2
        )
    
    return frame
    