"""
main.py — The Coordinator (The Pipeline Loop)

STORY METAPHOR:
In our restaurant story, this file is "The Coordinator". 
It acts as the director of the movie. 

Its job is to:
1. Open the video file.
2. Read the video frame-by-frame.
3. Skip frames if needed to keep things running fast (Temporal Decimation).
4. Send the frames to "The Spotter" (detection.py) and "The Rememberer" (tracking.py).
5. Pass the results to "The Brain" (logic.py) to update times and visit counts.
6. Draw colorful tables, boxes, IDs, and a HUD statistics box on the video.
7. Save the resulting video and write a final JSON summary report to disk.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import supervision as sv

import config
from detection import PersonDetector
from logic import RestaurantLogic
from tracking import PersonTracker, get_centroids


# ─────────────────────────────────────────────
# 🎨 ANNOTATION DRAWING HELPERS
# ─────────────────────────────────────────────

def draw_table_zones(frame: np.ndarray, alpha: float = config.ZONE_ALPHA) -> np.ndarray:
    """
    Draws the virtual fences (polygons) around the tables.
    We draw a solid semi-transparent filled polygon inside and a bright line border.
    """
    overlay = frame.copy()
    for tid, info in config.TABLE_ZONES.items():
        poly = info["polygon"]
        color = info["color"]
        
        # Fill polygon with color on the overlay image
        cv2.fillPoly(overlay, [poly], color)
        
        # Draw the solid boundary lines on the primary frame image
        cv2.polylines(frame, [poly], isClosed=True, color=color, thickness=2)
        
        # Calculate the center coordinate of the table polygon to place the text label (e.g. "T1")
        cx = int(poly[:, 0].mean())
        cy = int(poly[:, 1].mean())
        cv2.putText(frame, tid, (cx - 15, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        
    # Merge the semi-transparent overlay back into the primary frame
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    return frame


def draw_hud(frame: np.ndarray, logic: RestaurantLogic, timestamp: float) -> np.ndarray:
    """
    Draws a Heads-Up Display (HUD) statistics dashboard in the top-left corner.
    Shows the current elapsed video time, how long each table has been occupied, 
    how many visits have occurred, and if anyone is currently sitting there.
    """
    stats = logic.get_table_stats()
    y_start = 20
    line_h = 22

    # Draw a dark background panel block behind the text to make it easy to read
    panel_w = 340
    panel_h = len(stats) * line_h + 40
    panel = frame[y_start - 5: y_start + panel_h, 10: 10 + panel_w].copy()
    cv2.rectangle(frame, (10, y_start - 5), (10 + panel_w, y_start + panel_h),
                  (20, 20, 20), -1)
    cv2.addWeighted(panel, 0.4, frame[y_start - 5: y_start + panel_h, 10: 10 + panel_w],
                    0.6, 0, frame[y_start - 5: y_start + panel_h, 10: 10 + panel_w])

    # Draw Title Header text
    cv2.putText(frame, f"[t={timestamp:.1f}s]  Table Monitoring",
                (15, y_start + 12), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (220, 220, 220), 1, cv2.LINE_AA)

    # Loop through each table and draw its stats
    for i, (tid, stat) in enumerate(stats.items()):
        y = y_start + 30 + i * line_h
        
        # Format time into minutes and seconds
        occ_sec = int(stat.total_occupied_seconds)
        occ_min = occ_sec // 60
        occ_s   = occ_sec % 60
        currently = len(stat.current_occupants)
        
        # Text string: "T1: 2m15s | visits=1 | now=2"
        text = (f"{tid}: {occ_min}m{occ_s:02d}s | "
                f"visits={stat.waiter_visits} | "
                f"now={currently}")
        
        color = config.TABLE_ZONES[tid]["color"]
        cv2.putText(frame, text, (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    return frame


def draw_tracks(
    frame: np.ndarray,
    detections: sv.Detections,
    logic: RestaurantLogic,
) -> np.ndarray:
    """
    Draws bounding boxes around tracked people, labeled with their IDs 
    and their current state (e.g. WALKING, AT_TABLE, SERVING).
    """
    if len(detections) == 0:
        return frame

    track_states = logic.get_track_states()

    # Draw bounding boxes for each person in the detections list
    for i in range(len(detections)):
        tid = int(detections.tracker_id[i]) if detections.tracker_id is not None else -1
        xyxy = detections.xyxy[i].astype(int)
        x1, y1, x2, y2 = xyxy

        # Get their log record from the Brain to check if they are a customer or waiter
        rec = track_states.get(tid)
        is_waiter = rec.is_waiter if rec else False

        # Color coding: Blue/Cyan (BGR) for waiters, Green for customers
        box_color = (255, 100, 0) if is_waiter else (0, 200, 100)

        # Get their state label string
        if rec:
            if is_waiter:
                state_label = rec.waiter_state.name
            else:
                state_label = rec.customer_state.name
        else:
            state_label = "?"

        # Draw the rectangle box around the person
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

        # Create label text e.g., "C5:AT_TABLE" (Customer 5) or "W2:SERVING" (Waiter 2)
        label = f"{'W' if is_waiter else 'C'}{tid}:{state_label}"
        
        # Draw background label tag box above their head
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(frame, (x1, y1 - lh - 6), (x1 + lw + 4, y1), box_color, -1)
        
        # Write text
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    return frame


# ─────────────────────────────────────────────
# 🚀 CORE PIPELINE LOOP
# ─────────────────────────────────────────────

def run_pipeline(
    video_path: str,
    output_path: str,
    show: bool = False,
    device: str = "cpu",
):
    """
    Runs the entire pipeline loop end-to-end on the video file.
    """
    # Open the video file stream using OpenCV
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open video: {video_path}")

    # Fetch details about the input video file
    fps        = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[Main] Video: {video_path}")
    print(f"[Main] Resolution: {width}x{height}  FPS: {fps:.1f}  Frames: {total_frames}")

    # Set up the video writer to save the annotated output video
    os.makedirs(Path(output_path).parent, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Initialize our helper modules (The Spotter, Rememberer, and Brain)
    detector = PersonDetector(device=device)
    tracker  = PersonTracker(frame_rate=int(fps))
    logic    = RestaurantLogic()

    frame_idx  = 0
    skip_n     = config.PROCESS_EVERY_N_FRAMES
    t_start    = time.time()
    last_dets  = sv.Detections.empty()   # Store tracking results to reuse when skipping frames

    print(f"[Main] Starting processing (every {skip_n} frame(s))...")
    print("[Main] Press Q in the display window to stop early.\n")

    try:
        while True:
            # Read next frame
            ret, frame = cap.read()
            if not ret:
                break # Reached the end of the video

            # Calculate video time elapsed in seconds
            timestamp = frame_idx / fps

            # --- OPTIMIZATION: FRAME SKIPPING ---
            # Instead of running heavy neural networks on every frame, we skip frames.
            # E.g., if skip_n is 2, we run YOLO only on frames 0, 2, 4... 
            # On frames 1, 3, 5... we reuse the last tracking results, which saves 50% CPU/GPU.
            if frame_idx % skip_n == 0:
                # 1. Run Detection (The Spotter finds people)
                dets = detector.detect(frame)

                # 2. Run Tracking (The Rememberer matches boxes to track IDs)
                tracked = tracker.update(dets)
                last_dets = tracked
            else:
                # Reuse last tracked positions (no model inference)
                tracked = last_dets

            # 3. Update Decision Logic (The Brain evaluates table positions and times)
            if len(tracked) > 0 and tracked.tracker_id is not None:
                # Compute centroid center points
                centroids = get_centroids(tracked)
                
                # Send tracking arrays and elapsed time to logic engine state machine
                logic.update(
                    track_ids=tracked.tracker_id,
                    centroids=centroids,
                    timestamp=timestamp,
                    fps=fps,
                )

            # 4. Draw Annotations on current frame
            annotated = frame.copy()

            if config.DRAW_ZONES:
                annotated = draw_table_zones(annotated)

            if config.DRAW_TRACKS:
                annotated = draw_tracks(annotated, tracked, logic)

            if config.DRAW_STATS:
                annotated = draw_hud(annotated, logic, timestamp)

            # 5. Write the annotated frame to output file
            writer.write(annotated)

            # Show live window if requested
            if show:
                cv2.imshow("Restaurant Monitor", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[Main] User quit early.")
                    break

            frame_idx += 1

            # Print terminal progress logs every 5 video seconds
            if frame_idx % int(fps * 5) == 0:
                elapsed = time.time() - t_start
                pct = frame_idx / max(total_frames, 1) * 100
                print(f"  [{pct:5.1f}%] frame={frame_idx}  t={timestamp:.1f}s  "
                      f"wall={elapsed:.1f}s  tracks={len(tracked)}")

    finally:
        # Finish: Close any active sitting session times at the very end
        logic.finalize(frame_idx / fps)
        
        # Clean up files and windows
        cap.release()
        writer.release()
        if show:
            cv2.destroyAllWindows()

    # Save session summary metrics log as a JSON file
    summary = logic.get_summary()
    json_path = str(Path(output_path).parent / "session_log.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print a beautiful final report to console terminal
    print("\n" + "=" * 60)
    print("📊  FINAL REPORT")
    print("=" * 60)
    for tid, tdata in summary["tables"].items():
        name = config.TABLE_ZONES[tid]["name"]
        occ  = tdata["total_occupied_seconds"]
        vis  = tdata["waiter_visits"]
        m, s = divmod(int(occ), 60)
        print(f"  {name}")
        print(f"    → Occupied for:  {m}m {s:02d}s")
        print(f"    → Waiter visits: {vis}")
        print()

    waiters = summary.get("identified_waiters", [])
    print(f"  Auto-identified waiters (track IDs): {waiters}")
    print(f"\n  Annotated video : {output_path}")
    print(f"  JSON session log: {json_path}")
    print("=" * 60)

    return summary


# ─────────────────────────────────────────────
# 📥 CLI ARGUMENTS ENTRY POINT
# ─────────────────────────────────────────────

def parse_args():
    """
    Parses optional command line configuration inputs.
    """
    parser = argparse.ArgumentParser(
        description="Restaurant CCTV People Monitoring System"
    )
    parser.add_argument(
        "--video", "-v",
        default=config.VIDEO_PATH,
        help="Path to input video file (default: from config.py)",
    )
    parser.add_argument(
        "--output", "-o",
        default=os.path.join(config.OUTPUT_DIR, "annotated_output.mp4"),
        help="Path for annotated output video",
    )
    parser.add_argument(
        "--show", "-s",
        action="store_true",
        default=config.SHOW_LIVE,
        help="Display live annotated frames during processing",
    )
    parser.add_argument(
        "--device", "-d",
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Compute device for YOLOv8 (default: cpu)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        video_path=args.video,
        output_path=args.output,
        show=args.show,
        device=args.device,
    )
