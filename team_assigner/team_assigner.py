from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import cv2
import sys
sys.path.append(".../")
from utils import read_stub, save_stub


class TeamAssigner:
    def __init__(self, 
                team1_class_name = "white shirt", 
                team2_class_name = "dark blue shirt"):
        
        self.team1_class_name = team1_class_name
        self.team2_class_name = team2_class_name
        
        self.player_team_dict = {}
        
        
    def load_model(self, ):
        
        self.model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
        self.processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")


    def get_player_color(self, frame, bbox):
        # bbox = [x1, y1, x2, y2]
        '''
        frame is a NumPy array with shape (height, width, channels).

        To crop a region, we select:
            a range of rows (vertical axis → y),
            and a range of columns (horizontal axis → x).
        
        first slice (:) → rows (height)

        second slice (:) → columns (width)
        '''
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        
        # OpenCV uses BGR color order, while CLIP (via PIL) expects RGB.
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        
        classes = [self.team1_class_name, self.team2_class_name]
        
        inputs = self.processor(text=classes, images=pil_image, return_tensors="pt", padding=True)

        '''
        CLIP computes similarity scores between the image and each text prompt.
        Result example (for 1 image, 2 text labels):

        logits_per_image = tensor([[12.3, 9.5]])

        12.3 → similarity with “white shirt”

        9.5 → similarity with “dark blue shirt”
        If you have 1 image and 2 text prompts (“white shirt” and “dark blue shirt”), the shape is:

        logits_per_image.shape = (1, 2)

        Row	Meaning
        Axis 0 (dim=0)	image index (we only have 1)
        Axis 1 (dim=1)	text prompt index (one score per label)
        '''
        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image # this is the image-text similarity score
        
        '''
        Applies softmax → converts scores into normalized probabilities:

        probs = tensor([[0.90, 0.10]])

        Meaning:

        90 % chance → white shirt

        10 % chance → dark blue shirt.
        
        We apply softmax across the label dimension, so the two scores become probabilities that add up to 1 for each image.

        If we used dim=0, it would normalize across images (nonsense here, because we only have one image).
        So dim=1 = “normalize along the class axis.”
        '''
        probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
        
        '''
        probs.argmax(dim=1) → finds the index of the highest probability (0 or 1).

        probs is a PyTorch tensor of shape (N, M):
            Symbol	Meaning
                N	number of images in the batch
                M	number of text prompts (classes/labels)

        Example — if you have 1 image and 2 team names (“white shirt”, “dark blue shirt”):

        probs = tensor([[0.85, 0.15]])

        Shape = (1, 2)
        
        argmax(dim=1) finds the index of the maximum value along the specified dimension.

        dim=1 → along the columns, i.e., across the text classes for each image.

        Let’s compute:

        >>> probs.argmax(dim=1)
        tensor([0])


        Explanation:

        The largest value per row is in column index 0 (0.85 > 0.15).

        The result is a tensor of shape (N,), with one integer per image.

        If we had multiple images:

        probs = tensor([
            [0.85, 0.15],  # image 0
            [0.10, 0.90],  # image 1
        ])


        then:

        >>> probs.argmax(dim=1)
        tensor([0, 1])

        Image 0 → best match = index 0

        Image 1 → best match = index 1

        classes[index] → maps index back to text label.

        We have 1 image in the batch.
        So probs.argmax(dim=1) returns something like:

        tensor([0])
        That's a 1D tensor with a single element — not a plain Python integer.

        To extract the value (the actual scalar 0), you do [0].

        >>> probs.argmax(dim=1)[0]
        tensor(0)

        Now it's just a single scalar tensor, representing which class index is the predicted one for that image.

        We then use it as an index into our list of class names:

        class_name = classes[probs.argmax(dim=1)[0]]


        classes = ["white shirt", "dark blue shirt"]
        probs = tensor([[0.85, 0.15]])


        Compute step by step:
            1) probs.argmax(dim=1) → tensor([0])
            2) [0] → scalar 0
            3) classes[0] → "white shirt"

        Final output:
        class_name = "white shirt"
        
        Another example:
                probs = tensor([[0.25, 0.75]])
                argmax(dim=1) → tensor([1])

                [0] → 1

                classes[1] → "dark blue shirt"

                Result → "dark blue shirt"
        
        '''
        class_name = classes[probs.argmax(dim=1)[0]]
        return class_name
    
    def get_player_team(self, frame, player_bbox, player_id):
        
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]
        
        player_color = self.get_player_color(frame, player_bbox)
        
        team_id = 2
        if player_color == self.team1_class_name:
            team_id = 1
            
        self.player_team_dict[player_id] = team_id
        
        return team_id
        
    def get_player_teams_across_frames(self, video_frames, player_tracks, read_from_stub = False, stub_path = None):
        
        player_assignment = read_stub(read_from_stub, stub_path)
        if player_assignment is not None:
            if len(player_assignment) == len(video_frames):
                return player_assignment
        
        self.load_model()
        
        player_assignment = []
        
        for frame_num, player_track in enumerate(player_tracks):
            player_assignment.append({})
            
            # Every 50 frames i will clean the cache, so it has the opportunity to correct the wrong classifications
            if frame_num %50 == 0:
                self.player_team_dict={}
            
            for player_id, track in player_track.items():
                team = self.get_player_team(video_frames[frame_num],track["bbox"], player_id)
                player_assignment[frame_num][player_id] = team
        
        save_stub(stub_path,player_assignment)
        
        return player_assignment
                
                