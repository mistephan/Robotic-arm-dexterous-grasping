"""Open-vocabulary object detection.

Primary backend: Ultralytics YOLOE / YOLO-World (weights already present in
the baseline repo's models/ dir). Both accept free-text class prompts, which
is what lets the agent say "detect the red mug" for objects never seen in
training. A SAM3-style text-prompted segmenter can be slotted in behind the
same interface later (see docs/ROADMAP.md).
"""

from __future__ import annotations

import abc
import warnings

import numpy as np

from ..types import Detection, Frame


class Detector(abc.ABC):
    @abc.abstractmethod
    def detect(
        self,
        frame: Frame,
        classes: list[str] | None = None,
        imgsz: int | None = None,
    ) -> list[Detection]:
        """Run detection; `classes` overrides the prompt vocabulary."""

    @abc.abstractmethod
    def set_classes(self, classes: list[str]) -> None: ...


class OpenVocabDetector(Detector):
    def __init__(
        self,
        model_path: str,
        classes: list[str] | None = None,
        device: str = "cuda:0",
        conf: float = 0.25,
    ):
        # YOLOE currently loads its MobileCLIP text encoder from a
        # TorchScript .ts asset.  PyTorch emits a process-wide FutureWarning
        # for that supported upstream path; keep the console usable without
        # hiding unrelated warnings.  Removing TorchScript requires an
        # upstream Ultralytics/MobileCLIP format migration, not a local call
        # site change.
        warnings.filterwarnings(
            "ignore",
            message=r"`torch\.jit\.load` is deprecated\..*",
            category=FutureWarning,
            module=r"torch\.jit\._serialization",
        )
        from ultralytics import YOLO  # lazy heavy import

        self._model = YOLO(model_path)
        self._device = device
        self._conf = conf
        self._classes: list[str] = []
        self._filter_only = False  # closed-set model: filter by label instead
        if classes:
            self.set_classes(classes)

    def set_classes(self, classes: list[str]) -> None:
        if list(classes) == self._classes:
            return
        # YOLOE needs text embeddings passed explicitly; YOLO-World does not.
        # Closed-set models (yolo11n, ...) have no set_classes at all: fall
        # back to post-filtering detections by label (self._filter_only).
        self._filter_only = False
        try:
            self._model.set_classes(classes, self._model.get_text_pe(classes))
        except (AttributeError, TypeError):
            try:
                self._model.set_classes(classes)
            except (AttributeError, TypeError):
                self._filter_only = True
        except ModuleNotFoundError as e:
            raise RuntimeError(
                "open-vocabulary text prompts need ultralytics' CLIP fork: "
                "uv pip install git+https://github.com/ultralytics/CLIP.git "
                "(and 'mobileclip' for YOLOE models). Alternatively use a "
                f"closed-set model like yolo11n.pt. ({e})"
            ) from e
        self._classes = list(classes)

    def detect(
        self,
        frame: Frame,
        classes: list[str] | None = None,
        imgsz: int | None = None,
    ) -> list[Detection]:
        if classes:
            self.set_classes(classes)
        predict_kwargs = {
            "conf": self._conf,
            "device": self._device,
            "verbose": False,
        }
        if imgsz is not None:
            predict_kwargs["imgsz"] = int(imgsz)
        results = self._model.predict(frame.rgb, **predict_kwargs)
        dets = self._parse(results[0], frame)
        if self._filter_only and self._classes:
            # Loose word-overlap match so "red cup" still finds COCO's "cup".
            wanted = {w for c in self._classes for w in c.lower().split()}
            dets = [d for d in dets if wanted & set(d.label.lower().split())]
        return dets

    def _parse(self, result, frame: Frame) -> list[Detection]:
        dets: list[Detection] = []
        names = result.names
        boxes = result.boxes
        if boxes is None:
            return dets
        h, w = frame.rgb.shape[:2]
        masks = None
        if result.masks is not None:
            masks = result.masks.data.cpu().numpy()  # (N, mh, mw) float
        for i in range(len(boxes)):
            label = names[int(boxes.cls[i])]
            conf = float(boxes.conf[i])
            xyxy = boxes.xyxy[i].cpu().numpy().astype(np.float32)
            mask = None
            if masks is not None and i < len(masks):
                m = masks[i]
                if m.shape != (h, w):
                    import cv2

                    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                mask = m > 0.5
            obb = None
            if getattr(result, "obb", None) is not None and result.obb is not None:
                try:
                    obb = result.obb.xyxyxyxy[i].cpu().numpy().reshape(4, 2)
                except (IndexError, AttributeError):
                    obb = None
            dets.append(Detection(label=label, conf=conf, bbox=xyxy, mask=mask, obb=obb))
        return dets


class MockDetector(Detector):
    """Returns pre-configured detections; understands the synthetic scene.

    If constructed without explicit detections it auto-detects the red box in
    frames produced by `mock_camera.synthetic_tabletop` via color threshold,
    so end-to-end tests exercise real mask geometry.
    """

    def __init__(self, detections: list[Detection] | None = None, label: str = "red cube"):
        self._fixed = detections
        self._label = label
        self._classes: list[str] = []

    def set_classes(self, classes: list[str]) -> None:
        self._classes = list(classes)

    def detect(
        self,
        frame: Frame,
        classes: list[str] | None = None,
        imgsz: int | None = None,
    ) -> list[Detection]:
        if classes:
            self.set_classes(classes)
        if self._fixed is not None:
            return list(self._fixed)
        # Behave like an open-vocab detector: only "find" the object when the
        # requested vocabulary loosely matches its label.
        if self._classes:
            words = {w for c in self._classes for w in c.lower().split()}
            mine = set(self._label.lower().split())
            if not (words & mine):
                return []
        bgr = frame.rgb
        red = (
            (bgr[:, :, 2].astype(int) - bgr[:, :, 0].astype(int) > 60)
            & (bgr[:, :, 2].astype(int) - bgr[:, :, 1].astype(int) > 60)
        )
        if not red.any():
            return []
        ys, xs = np.nonzero(red)
        bbox = np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32)
        return [Detection(label=self._label, conf=0.95, bbox=bbox, mask=red)]
