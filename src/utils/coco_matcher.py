# src/core/coco_matcher.py
"""
Detect whether a dataset's class names overlap with COCO (YOLO-World's
high-confidence domain). Used to recommend the AI-assisted labeling default.

- High overlap  -> YOLO-World zero-shot pre-labeling is worth enabling.
- No overlap    -> niche dataset (cells, defects, ...) -> recommend manual.
"""
from src.config import COCO_CLASSES

# Normalized COCO vocabulary (lowercase, stripped) for fast matching
_COCO_NORM = [c.strip().lower() for c in COCO_CLASSES]
_COCO_SET = set(_COCO_NORM)

# Common synonyms / spelling variants → COCO canonical name
_SYNONYMS = {
    "people": "person",
    "persons": "person",
    "man": "person",
    "woman": "person",
    "human": "person",
    "bike": "bicycle",
    "automobile": "car",
    "auto": "car",
    "lorry": "truck",
    "van": "car",
    "aeroplane": "airplane",
    "plane": "airplane",
    "cell phone": "cell phone",
    "mobile phone": "cell phone",
    "telephone": "cell phone",
    "sofa": "couch",
    "tv monitor": "tv",
    "television": "tv",
    "dining table": "dining table",
    "table": "dining table",
    "pottedplant": "potted plant",
    "plant": "potted plant",
    "sportsball": "sports ball",
    "ball": "sports ball",
    "motorbike": "motorcycle",
    "truck": "truck",
    "stop sign": "stop sign",
    "trafficlight": "traffic light",
    "firehydrant": "fire hydrant",
    "parkingmeter": "parking meter",
    "handbag": "handbag",
    "suitcase": "suitcase",
    "frisbee": "frisbee",
    "snowboard": "snowboard",
    "surfboard": "surfboard",
    "tennisracket": "tennis racket",
    "wineglass": "wine glass",
    "hotdog": "hot dog",
    "potted plant": "potted plant",
    "diningtable": "dining table",
    "keyboard": "keyboard",
    "cellphone": "cell phone",
    "microwave": "microwave",
    "hairdrier": "hair drier",
    "hair dryer": "hair drier",
    "teddybear": "teddy bear",
    "toothbrush": "toothbrush",
}


def _normalize(name: str) -> str:
    """Lowercase + strip whitespace; resolve common synonyms."""
    n = name.strip().lower()
    n = _SYNONYMS.get(n, n)
    return n


def match_coco_overlap(class_names):
    """
    Compute overlap between dataset classes and the COCO vocabulary.

    Parameters
    ----------
    class_names : list[str]

    Returns
    -------
    (matched, total, ratio, matched_names)
        matched       : number of classes found in COCO (after synonym mapping)
        total         : total number of classes checked
        ratio         : matched / total (0.0 ~ 1.0)
        matched_names : list of COCO names that were matched
    """
    matched_names = []
    for name in class_names or []:
        n = _normalize(name)
        if not n:
            continue
        if n in _COCO_SET:
            matched_names.append(n)
    total = len([c for c in (class_names or []) if _normalize(c)])
    ratio = (len(matched_names) / total) if total else 0.0
    return matched_names, total, ratio


def recommend_ai_assist(class_names, ratio_threshold: float = 0.5) -> bool:
    """
    Recommend whether AI-assisted labeling should be ON by default.

    Rule: at least one matched class AND overlap ratio >= threshold.
    """
    matched, total, ratio = match_coco_overlap(class_names)
    return (len(matched) >= 1) and (ratio >= ratio_threshold)
