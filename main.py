from __future__ import annotations
import argparse, json, time
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
from intrusion_system import (
    load_polygon, point_in_polygon, foot_point, ZoneAlertTracker,
    draw_overlay, save_snapshot
)

def parse_source(value: str):
    return int(value) if value.isdigit() else value

def main():
    ap = argparse.ArgumentParser(description="CCTV intrusion detection and alert")
    ap.add_argument("--source", default="0", help="Camera index or video file/stream URL")
    ap.add_argument("--config", default="config/settings.json")
    ap.add_argument("--model", default=None, help="YOLO weights; overrides config")
    ap.add_argument("--output", default="runs/intrusion_output.mp4")
    ap.add_argument("--no-display", action="store_true")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    polygon = load_polygon(cfg["zone_polygon"])
    model_path = args.model or cfg.get("model", "yolov8n.pt")
    confidence = float(cfg.get("confidence", 0.35))
    classes = cfg.get("classes", [0])  # COCO class 0 = person
    fps_fallback = float(cfg.get("fps_fallback", 25))
    alert = ZoneAlertTracker(
        clear_delay_seconds=float(cfg.get("clear_delay_seconds", 4)),
        fps=fps_fallback,
        snapshot_cooldown_seconds=float(cfg.get("snapshot_cooldown_seconds", 3)),
    )
    model = YOLO(model_path)
    cap = cv2.VideoCapture(parse_source(args.source))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video source: {args.source}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps < 1 or fps > 120:
        fps = fps_fallback
    alert.set_fps(fps)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    snapshot_dir = Path(cfg.get("snapshot_dir", "snapshots"))
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    frame_no = 0

    print(f"[INFO] Model: {model_path} | Source: {args.source} | FPS: {fps:.1f}")
    print("[INFO] Press q to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_no += 1
            # Ultralytics persists ByteTrack IDs across frames.
            result = model.track(
                frame, persist=True, conf=confidence, classes=classes,
                tracker="bytetrack.yaml", verbose=False
            )[0]

            detections = []
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    conf = float(box.conf[0].cpu())
                    cls_id = int(box.cls[0].cpu())
                    track_id = int(box.id[0].cpu()) if box.id is not None else None
                    fx, fy = foot_point(x1, y1, x2, y2)
                    inside = point_in_polygon((fx, fy), polygon)
                    detections.append({
                        "box": (x1, y1, x2, y2), "confidence": conf,
                        "class_id": cls_id, "track_id": track_id,
                        "foot": (fx, fy), "intruding": inside
                    })

            state, intruder_count, should_snapshot = alert.update(detections)
            annotated = draw_overlay(frame, polygon, detections, state, intruder_count)
            if should_snapshot:
                saved = save_snapshot(annotated, snapshot_dir, frame_no)
                print(f"[ALERT] Intrusion detected: {intruder_count} object(s). Snapshot: {saved}")

            if writer is None:
                h, w = annotated.shape[:2]
                writer = cv2.VideoWriter(
                    str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
                )
            writer.write(annotated)
            if not args.no_display:
                cv2.imshow("Intrusion Detection and Alert", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
    print(f"[DONE] Annotated video: {output_path}")

if __name__ == "__main__":
    main()
