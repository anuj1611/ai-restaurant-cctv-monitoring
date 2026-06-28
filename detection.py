"""
detection.py — The Spotter

STORY METAPHOR:
In our restaurant story, this file is "The Spotter". 
Its only job is to stand at the door, look at a single frame of the security camera video, 
and point out where all the people are in that image. 

It does not remember anyone. If it looks at Frame 1 and then Frame 2, 
it does not know if a person in Frame 2 is the same person from Frame 1. 
It just spots them and writes down their location box coordinates.
"""

import numpy as np
import supervision as sv
from ultralytics import YOLO

import config


class PersonDetector:
    """
    Wraps the YOLOv8 model and provides a simple 'detect' function
    that takes an image and returns only the locations of detected people.
    """

    def __init__(
        self,
        model_path: str = config.YOLO_MODEL,
        confidence: float = config.DETECTION_CONFIDENCE,
        iou: float = config.DETECTION_IOU,
        classes: list = config.DETECT_CLASSES,
        device: str = "cpu",
    ):
        """
        Set up the Spotter.
        This runs once when the program starts. It loads the YOLO model weights from the disk.
        Loading the model from the disk takes time (disk I/O), so we do it here in __init__ 
        exactly once, rather than loading it inside the loop for every single frame.
        """
        print(f"[Detection] Loading YOLOv8 model: {model_path}  device={device}")
        
        # Load the pre-trained YOLOv8 neural network model
        self.model = YOLO(model_path)
        
        # Store our settings from config.py
        self.confidence = confidence  # How sure the AI must be to report a person
        self.iou = iou                # Threshold for removing duplicate boxes (NMS)
        self.classes = classes        # List of classes to detect (class 0 is Person)
        self.device = device          # Run on 'cpu' or 'cuda' (graphics card)

    def detect(self, frame: np.ndarray) -> sv.Detections:
        """
        Look at a single image frame (BGR format) and spot all the people.

        Parameters
        ----------
        frame : np.ndarray
            The image frame from the video (represented as a grid of pixel colors).

        Returns
        -------
        sv.Detections
            A list containing the results:
            - xyxy: The coordinate boxes around each person [left, top, right, bottom]
            - confidence: How sure the AI is about each person
            - class_id: The category index (0 for person)
        """
        # Run YOLOv8 model prediction on the frame
        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            iou=self.iou,
            classes=self.classes,
            device=self.device,
            verbose=False,            # Turn off printing YOLO status text to the terminal
        )[0]                          # Get the results for the first (and only) frame we passed

        # Convert the raw YOLO results into a standardized format using the 'supervision' library
        detections = sv.Detections.from_ultralytics(results)

        # Safety Check: Even though we requested only class 0 (person) in config, 
        # we filter again here to ensure we ONLY return detections that are labeled as class 0.
        if len(detections) > 0:
            mask = detections.class_id == 0
            detections = detections[mask]

        return detections
