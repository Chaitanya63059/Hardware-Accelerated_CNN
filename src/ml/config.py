"""
Shared detector configuration.

3-class subset: person, notebook (COCO 'book'), chair.
"""

TARGET_CLASSES = [
    {"name": "person", "coco_name": "person", "coco_id": 1},
]

CLASS_NAMES = [item["name"] for item in TARGET_CLASSES]
NUM_CLASSES = len(TARGET_CLASSES)
NUM_OUTPUTS = 5 + NUM_CLASSES  # [x, y, w, h, conf] + one-hot class vector

COCO_CATEGORY_ID_TO_CLASS_IDX = {
    item["coco_id"]: idx for idx, item in enumerate(TARGET_CLASSES)
}
