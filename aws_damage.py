# aws_damage.py — self-contained Rekognition + Annotate + CSV + PDF + S3
import os, io, csv, json
from datetime import datetime, timezone
from typing import Optional, Dict
import boto3
from PIL import Image, ImageDraw, ImageFont

print("[aws_damage] Loaded from:", __file__)

# ---------- Load AWS config ----------
with open("config_aws.json","r") as f:
    CFG = json.load(f)

REGION    = CFG.get("region", "us-east-2")
MODEL_ARN = CFG["model_arn"]
S3_BUCKET = CFG.get("s3_bucket", "")

rek = boto3.client("rekognition", region_name=REGION)
s3  = boto3.client("s3", region_name=REGION) if S3_BUCKET else None

# ---------- Utilities ----------
COLOR = {"minor": (255,193,7), "moderate": (255,87,34), "severe": (220,53,69)}

def _ensure_dir(p): os.makedirs(p, exist_ok=True)

def _pick_font(size):
    for p in [
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()

# ---------- Step 1: Rekognition detect ----------
def detect_severity(image_path: str, min_conf: float = 20.0) -> Dict:
    with open(image_path, "rb") as f:
        img_bytes = f.read()
    resp = rek.detect_custom_labels(
        ProjectVersionArn=MODEL_ARN,
        Image={"Bytes": img_bytes},
        MinConfidence=min_conf
    )
    labels = sorted(resp.get("CustomLabels", []), key=lambda x: x.get("Confidence", 0), reverse=True)
    if labels:
        return {"name": labels[0]["Name"], "conf": float(labels[0]["Confidence"]), "all": labels}
    else:
        return {"name": "NO_LABEL", "conf": 0.0, "all": []}

# ---------- Step 2: Annotate banner + boxes ----------
def annotate_with_banner(src_path: str, label: str, conf: float, out_path: str):
    with Image.open(src_path).convert("RGB") as im:
        W, H = im.size
        banner_h = max(40, H//16)
        draw = ImageDraw.Draw(im)
        color = COLOR.get(label.lower(), (0,123,255))
        draw.rectangle([(0,0),(W,banner_h)], fill=color)
        draw.text((10, banner_h//4), f"{label.upper()} ({conf:.1f}%)", fill=(255,255,255))
        im.save(out_path, quality=92)

# ---------- Step 3: CSV log ----------
def append_csv(log_csv: str, image: str, label: str, conf: float, labels_all):
    new = not os.path.exists(log_csv)
    with open(log_csv, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["timestamp_utc","image","top_label","top_confidence","all_labels"])
        w.writerow([
            datetime.now(timezone.utc).isoformat(),
            image,
            label,
            round(conf,2),
            "; ".join([f"{l['Name']}:{round(l['Confidence'],1)}" for l in labels_all])
        ])

# ---------- Step 4: Make a professional one-page PDF (built-in) ----------
def make_damage_report_pdf(
    annotated_path, label, conf, *,
    out_dir="rek_results",
    project_title="AI-Driven Car Damage Detection + ADAS",
    student="Chaitanya",
    university="Gannon University",
    course="MSc – Computer Information Science",
    model_note="AWS Rekognition Custom Labels (us-east-2)"
):
    _ensure_dir(out_dir)
    base = os.path.splitext(os.path.basename(annotated_path))[0].replace("_annotated","")
    out_pdf = os.path.join(out_dir, f"{base}_report.pdf")

    W, H = 1240, 1754
    pad = 60
    WHITE = (255,255,255); BLUE = (32,107,196); BLACK = (20,20,20)

    title_font = _pick_font(44)
    sub_font   = _pick_font(24)
    body_font  = _pick_font(22)
    small_font = _pick_font(20)

    page = Image.new("RGB", (W, H), WHITE)
    draw = ImageDraw.Draw(page)

    # header
    header_h = 110
    draw.rectangle([0,0,W,header_h], fill=BLUE)
    draw.text((pad, 28), "Vehicle Damage Assessment Report", fill=(255,255,255), font=title_font)
    draw.text((pad, 72), project_title, fill=(235,242,255), font=sub_font)

    # big image
    with Image.open(annotated_path).convert("RGB") as img:
        usable_h = H - (header_h + pad*3)
        usable_w = W - pad*2
        ratio = img.width / img.height
        target_h = int(usable_h * 0.92)
        target_w = int(target_h * ratio)
        if target_w > usable_w:
            target_w = usable_w
            target_h = int(target_w / ratio)
        img = img.resize((target_w, target_h))
        x = (W - target_w)//2
        y = header_h + pad*2
        draw.rectangle([x-6,y-6,x+target_w+6,y+target_h+6], outline=(210,210,210), width=2)
        page.paste(img, (x, y))

    # meta
    meta_y = header_h + pad*2 + target_h + 30
    draw.text((pad, meta_y), f"Detected: {label.upper()} ({conf:.1f}%)", fill=BLACK, font=body_font)
    draw.text((pad, meta_y+36), f"Timestamp (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}", fill=BLACK, font=small_font)
    draw.text((pad, meta_y+68), f"Student: {student}", fill=BLACK, font=small_font)
    draw.text((pad, meta_y+100), f"Course: {course}", fill=BLACK, font=small_font)
    draw.text((pad, meta_y+132), f"University: {university}", fill=BLACK, font=small_font)
    if model_note:
        draw.text((pad, meta_y+164), f"Model: {model_note}", fill=BLACK, font=small_font)

    # footer
    draw.text((pad, H-40), f"{university} • {course}", fill=(110,110,110), font=small_font)

    page.save(out_pdf, "PDF", resolution=300.0)
    return out_pdf

# ---------- Step 5: S3 upload ----------
def upload_to_s3(file_path: str, bucket: str, key: Optional[str] = None, expires_sec: int = 86400) -> Optional[str]:
    if not bucket or not s3:
        return None
    if key is None:
        key = os.path.basename(file_path)
    s3.upload_file(file_path, bucket, key)
    url = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=expires_sec
    )
    return url

# ---------- Step 6: Full pipeline ----------
def analyze_make_report(image_path: str, out_dir: str = "rek_results", min_conf: float = 20.0, upload: bool = False):
    """Detect → annotated JPG → CSV → PDF → (optional) S3 upload. Returns paths and URLs."""
    _ensure_dir(out_dir)

    result = detect_severity(image_path, min_conf=min_conf)
    label, conf = result["name"], result["conf"]

    base = os.path.splitext(os.path.basename(image_path))[0]
    annotated = os.path.join(out_dir, f"{base}_annotated.jpg")
    annotate_with_banner(image_path, label, conf, annotated)

    log_csv = os.path.join(out_dir, "damage_severity_log.csv")
    append_csv(log_csv, image_path, label, conf, result["all"])

    pdf_path = make_damage_report_pdf(annotated, label, conf, out_dir=out_dir)

    s3_links = {}
    if upload and S3_BUCKET:
        s3_links["annotated_url"] = upload_to_s3(annotated, S3_BUCKET, key=f"reports/{os.path.basename(annotated)}")
        s3_links["pdf_url"]       = upload_to_s3(pdf_path, S3_BUCKET, key=f"reports/{os.path.basename(pdf_path)}")

    return {
        "label": label,
        "confidence": conf,
        "annotated_image": annotated,
        "pdf_report": pdf_path,
        "csv_log": log_csv,
        "s3_links": s3_links
    }