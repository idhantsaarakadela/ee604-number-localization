
import cv2
import numpy as np
from ultralytics import YOLO


class NumericalTextDetector:

    """
    SVHN house-number detection pipeline.

    Assumption: Each input image contains one house-number sequence.
    Output coordinates use original-image pixels (x1, y1, x2, y2).
    """

    def __init__(self, model_path, confidence=0.25, imgsz=640):

        self.model = YOLO(model_path)
        self.confidence = confidence
        self.imgsz = imgsz

    def detect_digits(self, image):

        if isinstance(image, str):
            image = cv2.imread(image)

        if image is None:
            raise ValueError("Could not read input image.")

        height, width = image.shape[:2]

        results = self.model.predict(
            source=image,
            imgsz=self.imgsz,
            conf=self.confidence,
            iou=0.65,
            verbose=False
        )

        detections = []

        for result in results:

            if result.boxes is None:
                continue

            for box in result.boxes:

                digit = int(box.cls.item())
                confidence = float(box.conf.item())

                x1, y1, x2, y2 = [
                    float(v) for v in box.xyxy[0].tolist()
                ]

                # Ensure coordinates remain inside the image
                x1 = max(0.0, min(x1, width))
                x2 = max(0.0, min(x2, width))
                y1 = max(0.0, min(y1, height))
                y2 = max(0.0, min(y2, height))

                detections.append({
                    "digit": digit,
                    "confidence": confidence,
                    "bbox": [x1, y1, x2, y2]
                })

        return detections

    @staticmethod
    def group_digits(detections):

        """
        Reconstruct one number by sorting detected digits
        from left to right.

        Appropriate for the single-number-per-image SVHN setting.
        """

        if not detections:
            return []

        ordered = sorted(
            detections,
            key=lambda d: (
                d["bbox"][0] + d["bbox"][2]
            ) / 2
        )

        number = "".join(
            str(d["digit"]) for d in ordered
        )

        x1 = min(d["bbox"][0] for d in ordered)
        y1 = min(d["bbox"][1] for d in ordered)
        x2 = max(d["bbox"][2] for d in ordered)
        y2 = max(d["bbox"][3] for d in ordered)

        confidence = float(np.mean([
            d["confidence"] for d in ordered
        ]))

        return [{
            "text": number,
            "bbox": [x1, y1, x2, y2],
            "confidence": confidence
        }]

    def predict(self, image):

        if isinstance(image, str):
            image = cv2.imread(image)

        if image is None:
            raise ValueError("Could not read input image.")

        digits = self.detect_digits(image)
        sequences = self.group_digits(digits)

        return {
            "digits": digits,
            "sequences": sequences
        }

    @staticmethod
    def draw_predictions(image, result):

        output = image.copy()

        # Individual digit boxes: blue
        for detection in result["digits"]:

            x1, y1, x2, y2 = map(
                int, detection["bbox"]
            )

            label = str(detection["digit"])

            cv2.rectangle(
                output, (x1, y1), (x2, y2),
                (255, 100, 0), 1
            )

            cv2.putText(
                output, label,
                (x1, max(15, y1 - 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (255, 100, 0), 1
            )

        # Complete number box: green
        for sequence in result["sequences"]:

            x1, y1, x2, y2 = map(
                int, sequence["bbox"]
            )

            label = (
                f'{sequence["text"]} '
                f'({sequence["confidence"]:.2f})'
            )

            cv2.rectangle(
                output, (x1, y1), (x2, y2),
                (0, 255, 0), 2
            )

            cv2.putText(
                output, label,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 0), 2
            )

        return output
