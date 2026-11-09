from __future__ import annotations
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np

def load_polygon(points):
    """Convert configured [{x,y}, ...] points into OpenCV polygon format."""
    if len(points) < 3:
        raise ValueError("A zone polygon needs at least three points.")
    return np.asarray([[int(p["x"]), int(p["y"])] for p in points], dtype=np.int32).reshape((-1, 1, 2))

def point_in_polygon(point, polygon) -> bool:
    """Return True when point is inside or on the boundary of the zone."""
    return cv2.pointPolygonTest(polygon, (float(point[0]), float(point[1])), False) >= 0

def foot_point(x1, y1, x2, y2):
    """Bottom-centre of the detection box, used as ground-contact point."""
    return ((x1 + x2) // 2, y2)

class ZoneAlertTracker:
    """Frame-count based alert state machine; holds ALERT briefly after exit."""
    def __init__(self, clear_delay_seconds=4.0, fps=25.0, snapshot_cooldown_seconds=3.0):
        self.clear_delay_seconds = float(clear_delay_seconds)
        self.snapshot_cooldown_seconds = float(snapshot_cooldown_seconds)
        self.set_fps(fps)
        self.clear_frames = 0
        self.cooldown_frames = 0
        self.state = "SAFE"

    def set_fps(self, fps):
        self.fps = max(1.0, float(fps))
        self.clear_delay_frames = max(1, int(self.clear_delay_seconds * self.fps))
        self.snapshot_cooldown_frames = max(1, int(self.snapshot_cooldown_seconds * self.fps))

    def update(self, detections):
        intruders = sum(1 for d in detections if d.get("intruding", False))
        was_safe = self.state == "SAFE"
        if intruders:
            self.state = "ALERT"
            self.clear_frames = self.clear_delay_frames
        elif self.clear_frames > 0:
            self.clear_frames -= 1
            self.state = "ALERT"
        else:
            self.state = "SAFE"

        take_snapshot = False
        if self.cooldown_frames > 0:
            self.cooldown_frames -= 1
        if intruders and (was_safe or self.cooldown_frames == 0):
            take_snapshot = True
            self.cooldown_frames = self.snapshot_cooldown_frames
        return self.state, intruders, take_snapshot

def draw_overlay(frame, polygon, detections, state, intruder_count):
    out = frame.copy()
    h, w = out.shape[:2]
    # Semi-transparent zone fill, with amber border.
    overlay = out.copy()
    fill = (0, 180, 0) if state == "SAFE" else (0, 0, 220)
    cv2.fillPoly(overlay, [polygon], fill)
    out = cv2.addWeighted(overlay, 0.16, out, 0.84, 0)
    border = (0, 190, 255)
    cv2.polylines(out, [polygon], True, border, 2, cv2.LINE_AA)

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        inside = det["intruding"]
        color = (0, 0, 230) if inside else (80, 210, 80)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        fx, fy = det["foot"]
        cv2.circle(out, (fx, fy), 4, (255, 180, 0), -1)
        tid = det.get("track_id")
        label = f"PERSON {tid if tid is not None else '-'} {det['confidence']:.2f}"
        cv2.putText(out, label, (x1, max(18, y1-8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

    is_alert = state == "ALERT"
    banner_color = (0, 0, 220) if is_alert else (0, 135, 0)
    text = f"ALERT! {intruder_count} intruder(s)" if is_alert else "STATUS: SAFE"
    cv2.rectangle(out, (8, 8), (min(w-8, 360), 46), banner_color, -1)
    cv2.putText(out, text, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72,
                (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(out, "INTRUSION DETECTION", (10, h-14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
    return out

def save_snapshot(frame, directory, frame_no):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = directory / f"intrusion_{stamp}_frame{frame_no}.jpg"
    cv2.imwrite(str(path), frame)
    return path
