"""Object and person detection using YOLOv8."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from ultralytics import YOLO


@dataclass
class ObjectResult:
    person_count: int
    phone_detected: bool
    detected_labels: List[str]


class ObjectMonitor:
    """Runs lightweight YOLOv8 inference and extracts relevant classes."""

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        conf_threshold: float = 0.5,
        yolo_every_n_frames: int = 3,
        person_conf_threshold: float = 0.55,
        person_min_area_ratio: float = 0.03,
        phone_conf_threshold: float = 0.55,
    ) -> None:
        self.model = YOLO(model_name)
        self.conf_threshold = conf_threshold
        self.yolo_every_n_frames = max(1, yolo_every_n_frames)
        self.person_conf_threshold = person_conf_threshold
        self.person_min_area_ratio = person_min_area_ratio
        self.phone_conf_threshold = phone_conf_threshold
        self.frame_counter = 0
        self._last_result = ObjectResult(
            person_count=0,
            phone_detected=False,
            detected_labels=[],
        )

    def process_frame(self, frame_bgr: np.ndarray) -> ObjectResult:
        self.frame_counter += 1
        if self.frame_counter % self.yolo_every_n_frames != 0:
            return self._last_result

        result = self.model.predict(
            source=frame_bgr,
            conf=self.conf_threshold,
            verbose=False,
            imgsz=640,
            device="cpu",
        )[0]

        person_count = 0
        phone_detected = False
        labels: List[str] = []
        frame_h, frame_w = frame_bgr.shape[:2]
        frame_area = float(frame_h * frame_w)

        if result.boxes is not None and len(result.boxes) > 0:
            classes = result.boxes.cls.tolist()
            confs = result.boxes.conf.tolist()
            boxes = result.boxes.xyxy.tolist()
            for cls_id, conf, box in zip(classes, confs, boxes):
                label = result.names[int(cls_id)]
                labels.append(label)
                if label == "person":
                    x1, y1, x2, y2 = box
                    box_area = max(0.0, (x2 - x1) * (y2 - y1))
                    box_area_ratio = box_area / (frame_area + 1e-6)
                    if (
                        conf >= self.person_conf_threshold
                        and box_area_ratio >= self.person_min_area_ratio
                    ):
                        person_count += 1
                if label in {"cell phone", "mobile phone", "phone"}:
                    if conf >= self.phone_conf_threshold:
                        phone_detected = True

        self._last_result = ObjectResult(
            person_count=person_count,
            phone_detected=phone_detected,
            detected_labels=labels,
        )
        return self._last_result
