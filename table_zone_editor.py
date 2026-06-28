"""
table_zone_editor.py — The Mapmaker

STORY METAPHOR:
In our restaurant story, this file is "The Mapmaker". 
Before the pipeline starts watching the video, we must define exactly where the tables are. 

This is a GUI utility tool that opens the reference image (sample.png), 
and lets the engineer click to draw polygons (invisible fences) around each table. 
Once drawn, it writes these coordinates directly back into the "Rule Book" (config.py).
"""

from __future__ import annotations

import ast
import re
import sys
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np

# ─────────────────────────────────────────────
# 📥 LOAD CURRENT bluePRINT CONFIGS
# ─────────────────────────────────────────────
try:
    import config
    TABLE_ZONES = deepcopy(config.TABLE_ZONES)
    IMAGE_PATH = "sample.png"
except ImportError:
    sys.exit("[ERROR] config.py not found. Run from the project root directory.")


# ─────────────────────────────────────────────
# 🖱️ MOUSE INTERACTION STATE VARIABLES
# ─────────────────────────────────────────────
current_points: list[tuple[int, int]] = []
editing_done   = False


def mouse_callback(event, x, y, flags, param):
    """
    Triggers whenever the user clicks or moves the mouse in the GUI window.
    """
    global current_points
    
    # Left Click adds a corner vertex coordinate to our active fence polygon
    if event == cv2.EVENT_LBUTTONDOWN:
        current_points.append((x, y))
        
    # Right Click acts as an UNDO, removing the last point added
    elif event == cv2.EVENT_RBUTTONDOWN:
        if current_points:
            current_points.pop()


# ─────────────────────────────────────────────
# 🎨 GUI RENDERING LOGIC
# ─────────────────────────────────────────────

def draw_state(base_img: np.ndarray, table_id: str, points: list, defined_zones: dict) -> np.ndarray:
    """
    Renders the current visual status on top of the reference image.
    Draws existing zones, the active editing zone, mouse clicks, and HUD instructions.
    """
    frame = base_img.copy()

    # Draw previously defined zones (drawn faded so they don't distract the user)
    for tid, info in defined_zones.items():
        poly = info["polygon"]
        color = info["color"]
        overlay = frame.copy()
        cv2.fillPoly(overlay, [poly], color)
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
        cv2.polylines(frame, [poly], True, color, 2)
        cx, cy = int(poly[:, 0].mean()), int(poly[:, 1].mean())
        cv2.putText(frame, tid, (cx - 10, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)

    # Draw the current active polygon we are editing right now
    color = TABLE_ZONES[table_id]["color"]
    if len(points) >= 2:
        pts = np.array(points, dtype=np.int32)
        cv2.polylines(frame, [pts], isClosed=False, color=color, thickness=2)
    if len(points) >= 3:
        pts = np.array(points, dtype=np.int32)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)

    # Draw small circle dots at each mouse click point
    for pt in points:
        cv2.circle(frame, pt, 5, color, -1)
        cv2.circle(frame, pt, 5, (255, 255, 255), 1)

    # Draw instructions list text overlay
    instructions = [
        f"Editing: {table_id} ({TABLE_ZONES[table_id]['name']})",
        "Left Click: add vertex | Right Click: undo vertex",
        "N: save & next table | R: reset | S: save all | Q: quit",
        f"Vertices: {len(points)}  (need >= 3)",
    ]
    y = 20
    for line in instructions:
        cv2.putText(frame, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 255), 1, cv2.LINE_AA)
        y += 22

    return frame


# ─────────────────────────────────────────────
# 💾 UPDATE RULE BOOK CONFIGS
# ─────────────────────────────────────────────

def save_zones_to_config(zones: dict, config_path: str = "config.py"):
    """
    Saves updated table zones back into config.py by replacing the 
    TABLE_ZONES variable block using regular expressions.
    """
    with open(config_path, "r") as f:
        content = f.read()

    # Construct the replacement string block for TABLE_ZONES variable dictionary
    lines = ["TABLE_ZONES = {\n"]
    for tid, info in zones.items():
        poly = info["polygon"].tolist()
        color = info["color"]
        lines.append(f'    "{tid}": {{\n')
        lines.append(f'        "name": "{info["name"]}",\n')
        lines.append(f'        "polygon": np.array(\n')
        lines.append(f'            {poly},\n')
        lines.append(f'            dtype=np.int32),\n')
        lines.append(f'        "color": {color},\n')
        lines.append(f'    }},\n')
    lines.append("}\n")

    new_block = "".join(lines)

    # Use regex to find "TABLE_ZONES = { ... }" pattern and swap it with the new_block
    pattern = r"TABLE_ZONES\s*=\s*\{.*?\n\}"
    new_content = re.sub(pattern, new_block.rstrip(), content, flags=re.DOTALL)

    with open(config_path, "w") as f:
        f.write(new_content)

    print(f"[Editor] Saved updated TABLE_ZONES to {config_path}")


# ─────────────────────────────────────────────
# 🏁 APPLICATION INITIALIZER
# ─────────────────────────────────────────────

def main():
    global current_points

    import argparse
    parser = argparse.ArgumentParser(description="Interactive table zone editor")
    parser.add_argument("--image", default=IMAGE_PATH, help="Reference image path")
    args = parser.parse_args()

    # Load reference image using OpenCV
    base_img = cv2.imread(args.image)
    if base_img is None:
        sys.exit(f"[ERROR] Cannot load image: {args.image}")

    print(f"[Editor] Image: {args.image}  ({base_img.shape[1]}x{base_img.shape[0]})")
    print("[Editor] Starting interactive table zone editor...")

    table_ids = list(TABLE_ZONES.keys())
    defined_zones: dict = {}  # Store zones confirmed during this editor session

    # Setup the OpenCV GUI window
    cv2.namedWindow("Table Zone Editor", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Table Zone Editor", 1280, 720)
    cv2.setMouseCallback("Table Zone Editor", mouse_callback)

    idx = 0
    while idx < len(table_ids):
        tid = table_ids[idx]
        
        # Load the existing coordinates as points to start editing
        current_points = list(map(tuple, TABLE_ZONES[tid]["polygon"].tolist()))
        print(f"\n[Editor] Now editing: {tid} — {TABLE_ZONES[tid]['name']}")

        while True:
            # Draw frame
            frame = draw_state(base_img, tid, current_points, defined_zones)
            cv2.imshow("Table Zone Editor", frame)
            
            # Read key presses (wait 20 milliseconds)
            key = cv2.waitKey(20) & 0xFF

            # Key "N" saves current zone and advances to editing the next table
            if key == ord("n") or key == ord("N"):
                if len(current_points) >= 3:
                    TABLE_ZONES[tid]["polygon"] = np.array(current_points, dtype=np.int32)
                    defined_zones[tid] = TABLE_ZONES[tid]
                    print(f"  Saved {tid}: {current_points}")
                    idx += 1
                    break
                else:
                    print(f"  [!] Need at least 3 vertices. Currently: {len(current_points)}")

            # Key "R" resets/clears active points so you can start drawing this table again
            elif key == ord("r") or key == ord("R"):
                current_points = []
                print(f"  Reset {tid}")

            # Key "S" saves all progress immediately and exits the program
            elif key == ord("s") or key == ord("S"):
                if len(current_points) >= 3:
                    TABLE_ZONES[tid]["polygon"] = np.array(current_points, dtype=np.int32)
                    defined_zones[tid] = TABLE_ZONES[tid]
                # Keep remaining unchanged tables as-is
                for remaining_tid in table_ids[idx + 1:]:
                    defined_zones[remaining_tid] = TABLE_ZONES[remaining_tid]
                
                save_zones_to_config(TABLE_ZONES)
                cv2.destroyAllWindows()
                print("[Editor] All zones saved. Exiting.")
                return

            # Key "Q" exits immediately without saving anything
            elif key == ord("q") or key == ord("Q"):
                print("[Editor] Quit without saving.")
                cv2.destroyAllWindows()
                return

    # Auto-save after completing all table polygons
    save_zones_to_config(TABLE_ZONES)
    cv2.destroyAllWindows()
    print("[Editor] All zones defined and saved.")


if __name__ == "__main__":
    main()
