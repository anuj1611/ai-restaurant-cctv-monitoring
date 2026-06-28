"""
tracking.py — The Rememberer

STORY METAPHOR:
In our restaurant story, this file is "The Rememberer".
Its job is to watch the boxes found by "The Spotter" (detection.py) across time. 

If the Spotter finds a person at coordinates A in Frame 1, and coordinates B in Frame 2, 
the Rememberer calculates if these boxes represent the same human being. 
If they match, it gives them a persistent tracking ID (like "Track ID 5"). 
This ID stays the same as long as the person is in the restaurant, 
allowing us to track how long they stay and where they walk.
"""

import numpy as np
import supervision as sv

import config


class PersonTracker:
    """
    Wraps the ByteTrack algorithm to assign and maintain unique ID numbers
    for each detected person across video frames.
    """

    def __init__(
        self,
        track_buffer: int = config.TRACK_BUFFER,
        track_thresh: float = config.TRACK_THRESH,
        match_thresh: float = config.MATCH_THRESH,
        frame_rate: int = 25,
    ):
        """
        Set up the Rememberer.
        """
        print(f"[Tracking] Initialising ByteTrack  buffer={track_buffer}  fps={frame_rate}")
        
        # Initialize ByteTrack.
        # - track_activation_threshold: Only trust detections above this confidence score.
        # - lost_track_buffer: How many frames to remember a lost person (default: 30 frames).
        # - minimum_matching_threshold: The IoU threshold for matching boxes.
        self.tracker = sv.ByteTrack(
            track_activation_threshold=track_thresh,
            lost_track_buffer=track_buffer,
            minimum_matching_threshold=match_thresh,
            frame_rate=frame_rate,
        )

    def update(self, detections: sv.Detections) -> sv.Detections:
        """
        Update the track positions using new detections from the Spotter.

        Parameters
        ----------
        detections : sv.Detections
            The list of new person bounding boxes spotted in this frame.

        Returns
        -------
        sv.Detections
            The same list of boxes, but now each box has a 'tracker_id' attached to it.
            If a detection cannot be matched and is invalid, it is removed.
        """
        if len(detections) == 0:
            return detections

        # ByteTrack calculates overlaps and movement velocities to assign track IDs
        tracked = self.tracker.update_with_detections(detections)
        return tracked

    def reset(self):
        """
        Clear all memories.
        Call this if you switch to processing a new video so that previous IDs don't carry over.
        """
        self.tracker.reset()


# ─────────────────────────────────────────────
# 📐 GEOMETRIC UTILITY FUNCTIONS
# ─────────────────────────────────────────────

def get_centroids(detections: sv.Detections) -> np.ndarray:
    """
    Find the geometric center point (x, y) of each person's bounding box.

    WHY WE USE CENTROIDS INSTEAD OF THE BOTTOM CENTER (FEET):
    - In autonomous cars, we track the feet (bottom center of a box) to see where a person stands on the ground.
    - In a restaurant, tables and chairs block the view of people's lower bodies.
    - The bottom of the bounding box might clip at their chest or waist level, which changes constantly.
    - The geometric center (the centroid) of the box is much more stable and acts as a better point 
      to test if a person is sitting inside the virtual table zones.

    Parameters
    ----------
    detections : sv.Detections
        Detections that have bounding boxes in [left, top, right, bottom] format.

    Returns
    -------
    np.ndarray
        A 2D array of shape [N, 2] containing the (x, y) coordinate for each person.
    """
    if len(detections) == 0:
        return np.empty((0, 2), dtype=np.float32)

    xyxy = detections.xyxy  # Get the box edges: left (0), top (1), right (2), bottom (3)
    
    # Calculate the horizontal center (cx) and vertical center (cy)
    cx = (xyxy[:, 0] + xyxy[:, 2]) / 2.0
    cy = (xyxy[:, 1] + xyxy[:, 3]) / 2.0
    
    # Merge cx and cy arrays into coordinate pairs [ [cx1, cy1], [cx2, cy2], ... ]
    return np.stack([cx, cy], axis=1).astype(np.float32)


def get_box_center(detections: sv.Detections) -> np.ndarray:
    """
    An alternative function name for calculating the bounding box geometric centers.
    Returns a 2D array of shape [N, 2].
    """
    if len(detections) == 0:
        return np.empty((0, 2), dtype=np.float32)

    xyxy = detections.xyxy
    cx = (xyxy[:, 0] + xyxy[:, 2]) / 2.0
    cy = (xyxy[:, 1] + xyxy[:, 3]) / 2.0
    return np.stack([cx, cy], axis=1).astype(np.float32)
