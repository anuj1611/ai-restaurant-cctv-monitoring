"""
config.py — The Rule Book & Map of the Diner

STORY METAPHOR:
In our restaurant story, this file is the "Rule Book" or the "Map". 
Every other helper module refers to this file to know:
- Where the tables are on the restaurant floor.
- How fast we should run.
- What rules/thresholds define a customer sitting at a table vs a waiter serving them.
"""

import numpy as np

# ─────────────────────────────────────────────
# 🎥 VIDEO INPUT & OUTPUT PATHS
# ─────────────────────────────────────────────
# This tells the coordinator where to find the security camera video file 
# and where to save the final files after processing.
VIDEO_PATH = "sample video.mp4"
OUTPUT_DIR = "output"

# ─────────────────────────────────────────────
# 🔍 OBJECT DETECTION CONFIGURATION
# ─────────────────────────────────────────────
# The Spotter (YOLOv8) uses these rules to detect people.

# The name of the YOLOv8 model file. 
# We use the 'nano' version ('yolov8n.pt') because it is the smallest and fastest, 
# allowing our pipeline to run quickly even on standard computers without a GPU.
YOLO_MODEL = "yolov8n.pt"

# The confidence threshold.
# This means: "Only report detections where the AI is at least 35% sure it saw a person."
# We keep this low (0.35) so we don't miss people when they are partially blocked 
# by tables, chairs, or other people (occlusion).
DETECTION_CONFIDENCE = 0.35

# Non-Maximum Suppression (NMS) IoU threshold.
# When the AI draws multiple overlapping boxes around a single person, 
# NMS cleans them up. An IoU of 0.45 means if two boxes overlap by 45% or more, 
# we merge them. This stops the AI from drawing duplicate boxes for the same person.
DETECTION_IOU = 0.45

# COCO dataset class index for a person.
# In the YOLO standard COCO model, index 0 is always "person".
# We filter out cars, cups, chairs, etc., and ONLY keep index 0.
DETECT_CLASSES = [0]

# ─────────────────────────────────────────────
# 👣 MULTI-OBJECT TRACKING CONFIGURATION
# ─────────────────────────────────────────────
# The Rememberer (ByteTrack) uses these rules to keep track of people's IDs.

# Track Buffer: How long (in frames) we remember someone after they disappear.
# At 25 frames-per-second, 30 frames is about 1.2 seconds.
# If a customer gets hidden behind a waiter for 1 second, we don't assign them 
# a new ID when they reappear; we remember them.
TRACK_BUFFER = 30

# Track Threshold: The confidence score needed to start a *brand new* track.
# We set this slightly higher than the detection confidence (0.4) to make sure 
# we don't start tracking random background noise.
TRACK_THRESH = 0.4

# Match Threshold: The limit for matching boxes between frames.
# If the distance or difference between boxes is too large (greater than 0.8 cost), 
# we do not match them, assuming it is a different person.
MATCH_THRESH = 0.8

# ─────────────────────────────────────────────
# ⏱️ STATE MACHINE THRESHOLDS (BUSINESS RULES)
# ─────────────────────────────────────────────
# The Brain (logic.py) uses these rules to decide when a state transition occurs.

# How many seconds a customer must continuously stand/sit inside a table zone 
# before we officially say they are sitting at that table ("AT_TABLE").
# This filters out people who are just walking past the table.
CUSTOMER_DWELL_SECONDS = 5.0

# How many seconds a waiter must stay inside a table zone to count as a "visit".
# This prevents counting a visit if a waiter just walks by to go to another area.
WAITER_SERVE_SECONDS = 3.0

# Grace period (seconds) before we assume a customer has left a table.
# If the camera temporarily loses a customer due to someone walking in front of them, 
# we wait 2.0 seconds before stopping their timer.
LEAVE_GRACE_SECONDS = 2.0

# ─────────────────────────────────────────────
# 🗺️ TABLE ZONES (THE DINER MAP)
# ─────────────────────────────────────────────
# Here we define the invisible fences (polygons) around each table.
# The coordinates are (x, y) pixel values calibrated for the video size (848 x 480).
# Each table has a name, a set of corner points, and a color to draw on the video.
TABLE_ZONES = {
    "T1": {
        "name": "Table 1 (Top-Left)",
        "polygon": np.array([
            [100, 115],
            [225, 115],
            [250, 215],
            [ 75, 215],
        ], dtype=np.int32),
        "color": (0, 255, 128),     # BGR color format (Greenish)
    },
    "T2": {
        "name": "Table 2 (Center-Left Solo)",
        "polygon": np.array([
            [270, 145],
            [390, 145],
            [410, 225],
            [250, 225],
        ], dtype=np.int32),
        "color": (0, 200, 255),     # Blue-ish
    },
    "T3": {
        "name": "Table 3 (Left Foreground)",
        "polygon": np.array([
            [100, 220],
            [260, 220],
            [280, 340],
            [ 80, 340],
        ], dtype=np.int32),
        "color": (255, 128, 0),     # Orange
    },
    "T4": {
        "name": "Table 4 (Center Large)",
        "polygon": np.array([
            [270, 270],
            [510, 270],
            [540, 430],
            [240, 430],
        ], dtype=np.int32),
        "color": (200, 0, 255),     # Purple
    },
    "T5": {
        "name": "Table 5 (Right-Center)",
        "polygon": np.array([
            [430, 175],
            [620, 175],
            [650, 310],
            [410, 310],
        ], dtype=np.int32),
        "color": (0, 128, 255),     # Orange-Blue
    },
    "T6": {
        "name": "Table 6 (Top-Right Group)",
        "polygon": np.array([
            [580, 115],
            [730, 115],
            [750, 220],
            [560, 220],
        ], dtype=np.int32),
        "color": (128, 255, 0),     # Lime Green
    },
    "T7": {
        "name": "Table 7 (Far-Right near Bar)",
        "polygon": np.array([
            [700, 200],
            [848, 200],
            [848, 380],
            [690, 380],
        ], dtype=np.int32),
        "color": (255, 0, 128),     # Pink/Magenta
    },
}

# ─────────────────────────────────────────────
# 👔 WAITER IDENTIFICATION RULES
# ─────────────────────────────────────────────
# How we identify a waiter based on how they walk.

# Manual Overrides: If we already know the tracking ID numbers of the waiters, 
# we can put them here (e.g. [3, 7]) to force the system to treat them as waiters.
WAITER_TRACK_IDS = []

# Enable behavior auto-detection.
# If True, the system will automatically try to guess who the waiters are.
WAITER_AUTO_DETECT = True

# How long (in seconds) we observe a person before we make a final decision on their role.
# 120.0 seconds = 2 minutes. We only use their first 2 minutes of behavior.
WAITER_OBSERVATION_WINDOW = 120.0

# How many distinct table zones the person must visit in that 2-minute window 
# to be classified as a waiter. Waiters visit many tables; customers usually visit only one.
WAITER_ZONE_VISITS_REQUIRED = 3

# ─────────────────────────────────────────────
# 🎨 VISUALIZATION SETTINGS
# ─────────────────────────────────────────────
# Rules for drawing boxes and stats on the final output video.
SHOW_LIVE = False           # If True, shows the video processing live in a window
DRAW_ZONES = True           # Draw the table fence outlines on the video
DRAW_TRACKS = True          # Draw bounding boxes and ID numbers around people
DRAW_STATE  = True          # Draw state labels above people (e.g., "WALKING", "AT_TABLE")
DRAW_STATS  = True          # Draw the HUD stats panel in the top-left corner

# Visual design tweaks
FONT_SCALE = 0.4
FONT_THICKNESS = 1
ZONE_ALPHA = 0.12           # How transparent the filled color inside the table fences is (0 = fully invisible, 1 = solid color)

# ─────────────────────────────────────────────
# ⚡ SPEED OPTIMIZATION
# ─────────────────────────────────────────────
# Process every Nth frame.
# 1 = Process all frames.
# 2 = Skip every second frame (process half). This cuts the AI computation load in half, 
#     making the code run twice as fast while keeping tracking quality high.
PROCESS_EVERY_N_FRAMES = 2
