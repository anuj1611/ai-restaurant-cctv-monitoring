"""
logic.py — The Brain (Geofencing & State Machine)

STORY METAPHOR:
In our restaurant story, this file is "The Brain". 
It takes the ID numbers and coordinates from "The Rememberer" (tracking.py) 
and checks them against the "Rule Book" (config.py). 

Its main tasks are:
1. Drawing invisible fences (polygons) around the tables.
2. Checking if a person's center point is inside a fence.
3. Guessing who is a customer and who is a waiter based on their walking habits (behavior model).
4. Running rules (State Machines) to calculate how long customers sit down 
   and how many times waiters serve a table, while ignoring quick walk-bys or temporary glitches.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from shapely.geometry import Point, Polygon

import config


# ─────────────────────────────────────────────
# 🚥 CUSTOMER & WAITER STATES
# ─────────────────────────────────────────────
# Enums are lists of words representing all the possible phases a person can be in.

class CustomerState(Enum):
    """
    Every Customer must be in exactly one of these phases at any time:
    - WALKING: Moving around the restaurant or walking to a table.
    - AT_TABLE: Sitting down at a table, eating or chatting.
    - LEFT: The customer was sitting at a table but has now walked out of view.
    """
    WALKING  = auto()
    AT_TABLE = auto()
    LEFT     = auto()


class WaiterState(Enum):
    """
    Every Waiter must be in exactly one of these phases at any time:
    - IDLE: Walking around, clean up, or not near any customer tables.
    - APPROACHING: Just stepped inside a table zone (starting service).
    - SERVING: Stayed at the table long enough to serve food (visit counted!).
    """
    IDLE        = auto()
    APPROACHING = auto()
    SERVING     = auto()


# ─────────────────────────────────────────────
# 📝 INDIVIDUAL TRACK RECORDS
# ─────────────────────────────────────────────

@dataclass
class TrackRecord:
    """
    This is like a "logbook page" for each person tracked by our system.
    We record their role, their current state, the table they are at, and historical events.
    """
    track_id: int

    # Role identification
    is_waiter: bool = False
    first_seen_time: float = 0.0           # Video timestamp (seconds) when we first saw this person

    # Customer state variables
    customer_state: CustomerState = CustomerState.WALKING
    current_table: Optional[str] = None    # The ID of the table they are currently at (e.g. "T3")
    table_entry_time: float = 0.0          # Timestamp when they first stepped inside the table zone
    table_confirmed_time: float = 0.0      # Timestamp when they were officially classified as "AT_TABLE"
    last_seen_time: float = 0.0            # Timestamp of the last frame where we saw this person

    # Waiter state variables
    waiter_state: WaiterState = WaiterState.IDLE
    waiter_table: Optional[str] = None     # The ID of the table they are currently serving
    waiter_zone_entry_time: float = 0.0    # Timestamp when they entered the table zone

    # Historical logging
    table_dwell_history: Dict[str, float] = field(default_factory=dict) # {table_id: total_seconds_spent}
    tables_visited_as_customer: Set[str] = field(default_factory=set)


# ─────────────────────────────────────────────
# 📊 TABLE STATS LOGGING
# ─────────────────────────────────────────────

@dataclass
class OccupancyEvent:
    """
    Records a single continuous sitting session at a table.
    """
    track_id: int
    start_time: float       # Video time (seconds) when sitting started
    end_time: float         # Video time (seconds) when sitting ended
    duration_seconds: float # Total duration of the sitting session


@dataclass
class TableStats:
    """
    Maintains the summary metrics for a specific table zone.
    """
    table_id: str
    occupancy_events: List[OccupancyEvent] = field(default_factory=list) # List of all sitting sessions
    waiter_visits: int = 0                                                # Total waiter visit counts
    current_occupants: Set[int] = field(default_factory=set)            # Track IDs currently sitting here

    @property
    def total_occupied_seconds(self) -> float:
        """Sum the durations of all sitting sessions recorded at this table."""
        return sum(e.duration_seconds for e in self.occupancy_events)

    def summary(self) -> dict:
        """Converts stats into a clean Python dictionary format for saving to JSON files."""
        return {
            "table_id": self.table_id,
            "waiter_visits": self.waiter_visits,
            "total_occupied_seconds": round(self.total_occupied_seconds, 1),
            "occupancy_events": [
                {
                    "track_id": e.track_id,
                    "start_time": round(e.start_time, 1),
                    "end_time": round(e.end_time, 1),
                    "duration_seconds": round(e.duration_seconds, 1),
                }
                for e in self.occupancy_events
            ],
        }


# ─────────────────────────────────────────────
# 🧠 MAIN LOGIC CLASS
# ─────────────────────────────────────────────

class RestaurantLogic:
    """
    The orchestrator of spatiotemporal reasoning rules.
    It takes frame updates and decides how people are interacting with tables over time.
    """

    def __init__(self):
        """
        Set up the Brain.
        """
        # Convert the polygon corner coordinates from config.py into Shapely Polygon objects.
        # This makes it easy to run "contains" calculations (point-in-polygon math).
        self.table_polygons: Dict[str, Polygon] = {
            tid: Polygon(info["polygon"].tolist())
            for tid, info in config.TABLE_ZONES.items()
        }

        # Dict to store the logbook (TrackRecord) of each tracked person: {track_id: TrackRecord}
        self.tracks: Dict[int, TrackRecord] = {}

        # Dict to store table statistics: {table_id: TableStats}
        self.table_stats: Dict[str, TableStats] = {
            tid: TableStats(table_id=tid)
            for tid in config.TABLE_ZONES
        }

        # Store the list of tables each ID has visited, to detect waiters behaviorally
        self._track_table_visit_counts: Dict[int, Set[str]] = defaultdict(set)

        # Active session tracker to measure duration: {track_id: (table_id, start_time)}
        self._active_occupancy: Dict[int, Tuple[str, float]] = {}

        # Fetch timing thresholds from config.py
        self._customer_dwell = config.CUSTOMER_DWELL_SECONDS # Needs 5 seconds of presence to seat
        self._waiter_serve   = config.WAITER_SERVE_SECONDS   # Needs 3 seconds of presence to serve
        self._grace          = config.LEAVE_GRACE_SECONDS    # 2 seconds buffer to handle tracking loss

    def update(
        self,
        track_ids: np.ndarray,          # List of track IDs visible in this frame
        centroids: np.ndarray,          # Coordinates of their geometric centers [N, 2]
        timestamp: float,               # Video time in seconds
        fps: float = 25.0,
    ):
        """
        Process the tracking details from a single video frame.
        """
        active_ids = set(track_ids.tolist()) if len(track_ids) else set()

        # ── Step A: Log new people ──
        for tid in active_ids:
            if tid not in self.tracks:
                # If we've never seen this ID before, write a new logbook page for them
                self.tracks[tid] = TrackRecord(
                    track_id=tid,
                    is_waiter=(tid in config.WAITER_TRACK_IDS), # Manual override check
                    first_seen_time=timestamp,
                    last_seen_time=timestamp,
                )

        # ── Step B: Update visible people ──
        for i, tid in enumerate(track_ids.tolist()):
            rec = self.tracks[tid]
            rec.last_seen_time = timestamp
            
            # Point representing the center of the person's bounding box
            pt = Point(float(centroids[i, 0]), float(centroids[i, 1]))

            # Check if this point is inside any table's virtual fence
            current_zone = self._zone_for_point(pt)

            # --- WAITER AUTO-DETECTION SYSTEM (Behavioral Logic) ---
            # If auto-detect is enabled and we don't know if they are a waiter yet:
            if config.WAITER_AUTO_DETECT and not rec.is_waiter:
                time_tracked = timestamp - rec.first_seen_time
                
                # We only observe their behavior during their first 2 minutes of appearance
                if time_tracked <= config.WAITER_OBSERVATION_WINDOW:
                    if current_zone:
                        # Record that they visited this table zone
                        self._track_table_visit_counts[tid].add(current_zone)
                    
                    # If they visited 3 or more different tables, they are a Waiter!
                    if len(self._track_table_visit_counts[tid]) >= config.WAITER_ZONE_VISITS_REQUIRED:
                        rec.is_waiter = True
                        print(f"[Logic] Track {tid} behaviorally inferred as Waiter "
                              f"(visited {len(self._track_table_visit_counts[tid])} distinct zones in {time_tracked:.1f}s)")

            # --- ROUTE TO STATE MACHINE ---
            # Waiters and Customers have different rules. Run the appropriate logic:
            if rec.is_waiter:
                self._update_waiter(rec, current_zone, timestamp)
            else:
                self._update_customer(rec, current_zone, timestamp)

        # ── Step C: Handle disappeared people ──
        # Find people we saw in the past but who are missing in the current frame
        missing_ids = set(self.tracks.keys()) - active_ids
        for tid in missing_ids:
            rec = self.tracks[tid]
            elapsed_since_seen = timestamp - rec.last_seen_time
            
            # If they have been missing for longer than the grace period (2.0s), 
            # we officially assume they have walked out of the scene.
            if elapsed_since_seen > self._grace:
                self._handle_track_lost(rec, timestamp)

    # ──────────────────────────────────────────
    # 🧑 CUSTOMER STATE MACHINE LOGIC
    # ──────────────────────────────────────────

    def _update_customer(self, rec: TrackRecord, zone: Optional[str], ts: float):
        """
        Updates the Customer state transitions frame-by-frame.
        """
        state = rec.customer_state

        # STATE 1: Customer is WALKING
        if state == CustomerState.WALKING:
            if zone is not None:
                if rec.current_table != zone:
                    # They just entered a new table fence. Record entry time.
                    rec.current_table = zone
                    rec.table_entry_time = ts
                else:
                    # They are staying inside the same table zone. Check how long they stayed.
                    dwell = ts - rec.table_entry_time
                    if dwell >= self._customer_dwell:
                        # TRANSITION: WALKING ──(after 5s)──> AT_TABLE
                        rec.customer_state = CustomerState.AT_TABLE
                        rec.table_confirmed_time = ts
                        self._open_occupancy(rec, rec.table_entry_time) # Start counting table occupancy
                        self.table_stats[zone].current_occupants.add(rec.track_id)
            else:
                # They are not in any table zone. Clear pending fields.
                rec.current_table = None

        # STATE 2: Customer is sitting AT_TABLE
        elif state == CustomerState.AT_TABLE:
            if zone == rec.current_table:
                # They are still sitting at the same table. Keep doing nothing.
                pass
            elif zone is None or zone != rec.current_table:
                # They stood up and walked away (or moved to another table zone)
                # TRANSITION: AT_TABLE ───> WALKING
                self._close_occupancy(rec, ts) # Stop counting table occupancy
                if rec.current_table in self.table_stats:
                    self.table_stats[rec.current_table].current_occupants.discard(rec.track_id)
                
                rec.customer_state = CustomerState.WALKING
                rec.current_table = zone
                if zone:
                    rec.table_entry_time = ts
                else:
                    rec.table_entry_time = 0.0

        # STATE 3: Customer LEFT the diner
        elif state == CustomerState.LEFT:
            # If they were marked as LEFT but reappear inside a zone, treat them as a new customer
            if zone is not None:
                rec.customer_state = CustomerState.WALKING
                rec.current_table = zone
                rec.table_entry_time = ts

    # ──────────────────────────────────────────
    # 👔 WAITER STATE MACHINE LOGIC
    # ──────────────────────────────────────────

    def _update_waiter(self, rec: TrackRecord, zone: Optional[str], ts: float):
        """
        Updates the Waiter state transitions frame-by-frame.
        """
        state = rec.waiter_state

        # STATE 1: Waiter is IDLE
        if state == WaiterState.IDLE:
            if zone is not None:
                # They just walked into a table zone
                # TRANSITION: IDLE ───> APPROACHING
                rec.waiter_state = WaiterState.APPROACHING
                rec.waiter_table = zone
                rec.waiter_zone_entry_time = ts

        # STATE 2: Waiter is APPROACHING table
        elif state == WaiterState.APPROACHING:
            if zone == rec.waiter_table:
                # Still inside the zone. Check how long they stayed.
                dwell = ts - rec.waiter_zone_entry_time
                if dwell >= self._waiter_serve:
                    # TRANSITION: APPROACHING ──(after 3s)──> SERVING (Visit counted!)
                    rec.waiter_state = WaiterState.SERVING
                    self.table_stats[zone].waiter_visits += 1
            else:
                # They walked past without serving (reset back to IDLE)
                rec.waiter_state = WaiterState.IDLE
                rec.waiter_table = zone
                if zone is not None:
                    rec.waiter_zone_entry_time = ts

        # STATE 3: Waiter is SERVING table
        elif state == WaiterState.SERVING:
            if zone != rec.waiter_table:
                # They completed the service and left the table zone
                # TRANSITION: SERVING ───> IDLE
                rec.waiter_state = WaiterState.IDLE
                rec.waiter_table = None

    # ──────────────────────────────────────────
    # ⏱️ OCCUPANCY TIMING HELPERS
    # ──────────────────────────────────────────

    def _open_occupancy(self, rec: TrackRecord, start_time: float):
        """Start recording a new sitting session for a customer."""
        if rec.track_id not in self._active_occupancy:
            self._active_occupancy[rec.track_id] = (rec.current_table, start_time)

    def _close_occupancy(self, rec: TrackRecord, end_time: float):
        """Stop recording and save the sitting session duration to the table log."""
        tid = rec.track_id
        if tid in self._active_occupancy:
            table_id, start_time = self._active_occupancy.pop(tid)
            duration = end_time - start_time
            if duration > 0 and table_id in self.table_stats:
                event = OccupancyEvent(
                    track_id=tid,
                    start_time=start_time,
                    end_time=end_time,
                    duration_seconds=duration,
                )
                self.table_stats[table_id].occupancy_events.append(event)

    def _handle_track_lost(self, rec: TrackRecord, ts: float):
        """
        Called when a track has been lost for longer than the grace period.
        It closes any open session timers and resets states.
        """
        # If they were sitting at a table, close their sitting session and remove them
        if rec.customer_state == CustomerState.AT_TABLE:
            self._close_occupancy(rec, ts)
            if rec.current_table in self.table_stats:
                self.table_stats[rec.current_table].current_occupants.discard(rec.track_id)
            rec.customer_state = CustomerState.LEFT

        # If they were a waiter serving or approaching, reset them to IDLE
        if rec.waiter_state in (WaiterState.APPROACHING, WaiterState.SERVING):
            rec.waiter_state = WaiterState.IDLE

    # ──────────────────────────────────────────
    # 🗺️ GEOFENCING MATH HELPER
    # ──────────────────────────────────────────

    def _zone_for_point(self, pt: Point) -> Optional[str]:
        """
        Uses Shapely's Point-in-Polygon check to find which table zone 
        contains the point.
        """
        for tid, poly in self.table_polygons.items():
            if poly.contains(pt):
                return tid # Return table ID (e.g. "T3")
        return None        # Return None if the point is in the open floor walkway
