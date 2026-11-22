"""YOLO -> InsightFace face assignment (FaceProcess 2.8, E3).

YOLO decides which faces a branch processes.  InsightFace only supplies, for
each already selected YOLO face, the matching complete face object (bbox,
five keypoints and the 106 landmarks the restore pasteback uses).  It never
selects a face of its own, and an unassigned YOLO face is skipped -- there is
no substitute face and no fallback to other faces.

Rule, calibrated on 2.8 detection data (60 synthetic scenes with ground truth,
188 real images, 80 scenes with strongly overlapping faces; 0 wrong
assignments with the rule, 3-4 without the keypoint condition):

- candidates are the InsightFace faces with at least MATCH_MIN_KEYPOINTS of
  their five keypoints inside the YOLO bbox; the best (highest IoU) needs
  IoU >= MATCH_IOU (otherwise UNMATCHED);
- it must lead the second candidate by at least MATCH_MARGIN IoU
  (otherwise AMBIGUOUS);
- an InsightFace face that is the best candidate of two YOLO faces is
  assigned to neither (CONFLICT).
"""

from __future__ import annotations

MATCH_RULE = "iou_kps_margin_v1"
MATCH_IOU = 0.30
MATCH_MIN_KEYPOINTS = 4
MATCH_MARGIN = 0.20

OK = "OK"
UNMATCHED = "UNMATCHED"
AMBIGUOUS = "AMBIGUOUS"
CONFLICT = "CONFLICT"


def _box(value):
    try:
        x1, y1, x2, y2 = [float(v) for v in list(value)[:4]]
    except Exception:
        return None
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def iou(a, b) -> float:
    a, b = _box(a), _box(b)
    if a is None or b is None:
        return 0.0
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def keypoints_inside(kps, box) -> int:
    box = _box(box)
    if box is None or kps is None:
        return 0
    count = 0
    for point in list(kps)[:5]:
        try:
            x, y = float(point[0]), float(point[1])
        except Exception:
            continue
        if box[0] <= x <= box[2] and box[1] <= y <= box[3]:
            count += 1
    return count


def assign_faces(yolo_boxes, faces):
    """Assign InsightFace faces to selected YOLO boxes.

    Returns one record per YOLO box, in input order:
    ``{"status", "face_index", "iou", "keypoints", "second_iou"}``.
    ``face_index`` is set only for OK.
    """
    candidates = []
    for box in yolo_boxes:
        row = []
        for index, face in enumerate(faces):
            overlap = iou(box, getattr(face, "bbox", None))
            inside = keypoints_inside(getattr(face, "kps", None), box)
            # The margin is measured against every keypoint-consistent
            # candidate (as calibrated), the IoU threshold applies to the best.
            if overlap > 0.0 and inside >= MATCH_MIN_KEYPOINTS:
                row.append((overlap, inside, index))
        candidates.append(sorted(row, reverse=True))

    records = []
    for row in candidates:
        if not row or row[0][0] < MATCH_IOU:
            records.append({"status": UNMATCHED, "face_index": None, "iou": 0.0, "keypoints": 0, "second_iou": 0.0})
            continue
        best = row[0]
        second = row[1][0] if len(row) > 1 else 0.0
        status = AMBIGUOUS if len(row) > 1 and best[0] - second < MATCH_MARGIN else OK
        records.append({"status": status, "face_index": best[2] if status == OK else None,
                        "iou": round(best[0], 4), "keypoints": best[1], "second_iou": round(second, 4)})

    claimed = {}
    for position, record in enumerate(records):
        if record["status"] == OK:
            claimed.setdefault(record["face_index"], []).append(position)
    for positions in claimed.values():
        if len(positions) > 1:
            for position in positions:
                records[position].update(status=CONFLICT, face_index=None)
    return records
