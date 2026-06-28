# 🍽️ AI Restaurant CCTV Monitoring

An edge-computational computer vision pipeline that analyzes restaurant CCTV feeds to track customer dwell times and monitor waiter service efficiency.

This system turns raw security camera footage into actionable business metrics without requiring expensive custom hardware or smart tables.

---

## ✨ Features

- **Customer Occupancy Tracking**: Calculates exactly how long customers sit at each table.
- **Waiter Service Monitoring**: Uses spatiotemporal behavioral inference to identify waiters and count how many times they visit and serve each table.
- **Robust Geofencing**: Allows interactive, point-and-click definition of table zones (virtual fences).
- **Edge-Optimized Pipeline**: Utilizes YOLOv8 (nano), ByteTrack, and temporal frame decimation to run efficiently on standard hardware.
- **State Machine Logic**: Deterministic tracking logic to handle real-world noise like bounding box jitter and temporary occlusions.

---

## 🏗️ System Architecture

The project is structured into modular components:

- `main.py` (The Coordinator): Handles video I/O, frame skipping, HUD rendering, and orchestrates the pipeline.
- `detection.py` (The Spotter): Runs **YOLOv8** to locate people in each frame.
- `tracking.py` (The Rememberer): Uses **ByteTrack** to assign persistent IDs to detected people, even through partial occlusions.
- `logic.py` (The Brain): Computes bounding box centroids, checks polygon collisions, and manages deterministic **State Machines** for tracking behavior.
- `config.py` (The Rule Book): Stores thresholds, hyperparameters, and table geofence coordinates.
- `table_zone_editor.py`: A visual GUI tool to redraw table geofences.

---

## ⚙️ How It Works

1. **Perception**: YOLOv8 predicts bounding boxes for people in the video frame.
2. **Association**: ByteTrack assigns persistent IDs to these boxes across consecutive frames.
3. **Geofencing**: The geometric centroid of each bounding box is calculated. A ray-casting algorithm checks if the centroid falls inside any predefined table zones.
4. **Behavioral State Machines**:
   - **Customers**: `WALKING` ➔ `AT_TABLE` (after a 5-second dwell time) ➔ `LEFT` (with a 2-second grace period for occlusions).
   - **Waiters**: Identified by visiting $\ge$ 3 distinct tables within a 2-minute window. State transitions: `IDLE` ➔ `APPROACHING` ➔ `SERVING` (after a 3-second dwell time).

---

## 🚀 Quick Start Guide

### Prerequisites
- Python 3.8+
- Git

### 1. Installation

Clone the repository and set up an isolated virtual environment:

```bash
git clone https://github.com/yourusername/ai-restaurant-cctv-monitoring.git
cd ai-restaurant-cctv-monitoring

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
.\venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Running the Pipeline

Run the main execution script. It reads the sample video, processes the AI pipeline, and displays the annotated tracking window live:

```bash
python main.py --video "sample video.mp4" --show
```
*(Processed output logs and annotated frames will be saved in the `output/` directory).*

### 3. Interactive Geofence Editor

If you want to map a new restaurant layout or redraw the boundary polygons for the tables, run the interactive editor against a reference image:

```bash
python table_zone_editor.py --image sample.png
```

**Editor Controls:**
* **Left Click**: Add a corner vertex point to the current table.
* **Right Click**: Undo the last point.
* **N**: Save the current zone and switch to drawing the next table.
* **S**: Save all zones and write coordinates back into `config.py`, then exit.

---

## 🛠️ Technology Stack
* **Computer Vision**: OpenCV, Ultralytics (YOLOv8)
* **Object Tracking**: Supervision (ByteTrack)
* **Geometry & Mathematics**: NumPy, Shapely
