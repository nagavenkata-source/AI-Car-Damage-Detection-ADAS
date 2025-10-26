# adas_live.py  — ADAS + Auto Rekognition + Auto S3 Upload on URGENT
import os
import time
import argparse
import cv2
import pyttsx3
from ultralytics import YOLO
from datetime import datetime, timezone

# ⬇️ NEW: our analysis + report + S3 uploader
# Requires: aws_damage.py + config_aws.json in the same folder
try:
    from aws_damage import analyze_make_report
    HAVE_AWS_HELPER = True
except Exception as e:
    print("[WARN] aws_damage not available:", e)
    HAVE_AWS_HELPER = False

# Classes we care about (COCO ids)
WATCH_CLASSES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

def speak(engine, text, enabled=True):
    if enabled and engine:
        try:
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print("TTS error:", e)

def approx_distance_ratio(bbox, H):
    """taller box → closer object (very simple proxy for distance)"""
    h = max(1, bbox[3] - bbox[1])
    return h / max(1, H)

def lateral_offset(bbox, W):
    """negative → left of center; positive → right of center"""
    cx = (bbox[0] + bbox[2]) / 2.0
    return (cx - W / 2) / (W / 2)

def analyze_and_upload_current_frame(frame, out_dir="rek_results", min_conf=20.0):
    """
    On-demand: save snapshot of current frame, analyze with Rekognition,
    create annotated JPG + PDF + CSV, auto-upload both to S3, print URLs.
    """
    if not HAVE_AWS_HELPER:
        print("[Rekognition/S3] Skipped: aws_damage.py not found (put it in project folder).")
        return

    os.makedirs(out_dir, exist_ok=True)
    snap_name = f"snapshot_{int(time.time())}.jpg"
    cv2.imwrite(snap_name, frame)
    print("[SNAPSHOT] Saved:", snap_name)

    try:
        res = analyze_make_report(
            snap_name,
            out_dir=out_dir,
            min_conf=min_conf,
            upload=True    # 🔥 auto-upload to S3
        )
        print(f"[Rekognition] {res['label'].upper()} ({res['confidence']:.1f}%)")
        print("[Local] Annotated:", res["annotated_image"])
        print("[Local] PDF:", res["pdf_report"])

        if res.get("s3_links"):
            print("[S3] Annotated URL:", res["s3_links"].get("annotated_url"))
            print("[S3] PDF URL:", res["s3_links"].get("pdf_url"))
            # One-line summary for your report
            print(
                "EVIDENCE:",
                res['label'].upper(), f"({res['confidence']:.1f}%)",
                "| IMG:", res['s3_links'].get('annotated_url'),
                "| PDF:", res['s3_links'].get('pdf_url'),
                "| UTC:", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            )

    except Exception as e:
        print("[Rekognition/S3] Error:", e)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="", help="Path to video; if empty, uses webcam 0")
    ap.add_argument("--conf", type=float, default=0.35, help="YOLO confidence threshold")
    ap.add_argument("--tts", action="store_true", help="Enable voice alerts")
    args = ap.parse_args()

    # Open input
    cap = cv2.VideoCapture(0 if args.video == "" else args.video)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video/webcam")

    # Prepare recorder (save full processed video)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    base = "webcam" if args.video == "" else os.path.splitext(os.path.basename(args.video))[0]
    output_name = f"{base}_processed.mp4"
    out = cv2.VideoWriter(output_name, fourcc, fps, (width, height))
    if not out.isOpened():
        raise RuntimeError("Cannot open VideoWriter for output file")
    print(f"[Recording] Saving processed video to {output_name}")

    # Load model + TTS
    model = YOLO("yolov8n.pt")  # downloads on first run
    tts_engine = pyttsx3.init() if args.tts else None

    last_level = "NONE"
    hysteresis = 6  # frames to confirm level change
    counter = 0

    print("ADAS running... Press Q to quit, S to save snapshot, U to upload snapshot to S3")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        H, W = frame.shape[:2]

        # Run detector (cars, people, etc.)
        res = model.predict(
            source=frame,
            conf=args.conf,
            classes=list(WATCH_CLASSES.keys()),
            verbose=False
        )

        dets = []
        for r in res:
            for b in r.boxes:
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
                cid = int(b.cls[0]); conf = float(b.conf[0])
                if cid in WATCH_CLASSES:
                    dets.append({"bbox": [x1, y1, x2, y2], "name": WATCH_CLASSES[cid], "conf": conf})

        # Pick "closest" object (largest height ratio)
        focus = None; best_ratio = 0.0
        for d in dets:
            ratio = approx_distance_ratio(d["bbox"], H)
            if ratio > best_ratio:
                best_ratio = ratio; focus = d

        # Decide risk (slightly easier to trigger for demos)
        level = "NONE"
        if focus:
            if best_ratio >= 0.35:
                level = "URGENT"
            elif best_ratio >= 0.20:
                level = "WARN"

        # Hysteresis to avoid chattering
        if level != last_level:
            counter += 1
            if counter >= hysteresis:
                last_level = level
                counter = 0
                if args.tts:
                    if last_level == "WARN":
                        off = lateral_offset(focus["bbox"], W) if focus else 0.0
                        tip = "Keep lane and slow down."
                        if off < -0.2: tip = "Hazard on left. Keep lane."
                        elif off > 0.2: tip = "Hazard on right. Keep lane."
                        speak(tts_engine, f"Caution. {tip}")
                    elif last_level == "URGENT":
                        speak(tts_engine, "Brake now. Possible collision ahead.")
                        # ⬇️ NEW: auto snapshot → analyze → PDF → CSV → S3
                        analyze_and_upload_current_frame(frame)
        else:
            counter = 0

        # Draw detections
        for d in dets:
            x1, y1, x2, y2 = map(int, d["bbox"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{d['name']} {d['conf']:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Banners
        if last_level == "URGENT":
            cv2.rectangle(frame, (0, 0), (W, 40), (0, 0, 255), -1)
            cv2.putText(frame, "URGENT: BRAKE NOW", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        elif last_level == "WARN":
            cv2.rectangle(frame, (0, 0), (W, 40), (0, 165, 255), -1)
            cv2.putText(frame, "CAUTION: VEHICLE AHEAD", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

        # Show + record
        cv2.imshow("ADAS - Software Only", frame)
        out.write(frame)  # save processed frame to the output video

        # Keys
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q")):
            break
        if key in (ord("s"), ord("S")):
            filename = f"snapshot_{int(time.time())}.jpg"
            cv2.imwrite(filename, frame)
            print("Saved snapshot:", filename)
            speak(tts_engine, "Snapshot saved.") if args.tts else None
        if key in (ord("u"), ord("U")):
            # ⬇️ NEW: on-demand snapshot → analyze → upload to S3
            analyze_and_upload_current_frame(frame)

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    if out:
        out.release()
        print(f"[Recording] Saved processed video to {output_name}")

if __name__ == "__main__":
    main()