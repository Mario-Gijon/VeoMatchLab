from __future__ import annotations

import gc
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image, ImageDraw, ImageFont


DEFAULT_THRESHOLD = 0.25
MODEL_FILENAME = "rf-detr-large-2026.pth"


@dataclass(frozen=True, slots=True)
class PersonDetection:
    """
    Resultado normalizado de una detección de persona.

    No contiene tracking, identidad, proyección al campo ni información
    temporal. Es únicamente la salida espacial del detector para un frame.
    """

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2, self.y2

    @property
    def bottom_center(self) -> tuple[float, float]:
        """
        Punto aproximado de contacto de la persona con el suelo.

        Todavía no se utiliza para proyección; simplemente se deja
        disponible para la siguiente capa.
        """
        return (self.x1 + self.x2) / 2.0, self.y2

    def to_dict(self) -> dict[str, float | list[float]]:
        return {
            "bbox": [self.x1, self.y1, self.x2, self.y2],
            "confidence": self.confidence,
            "bottom_center": list(self.bottom_center),
        }


class RFDETRLargePersonDetector:
    """
    Detector puro de personas usando RF-DETR Large.

    Responsabilidad:
        imagen -> detecciones de personas

    Expresamente NO hace:
        - tracking
        - IDs
        - temporalidad
        - clasificación jugador/árbitro
        - proyección al campo
        - deduplicación entre lentes
    """

    model_name = "RF-DETR Large"
    model_id = "rfdetr_large"

    def __init__(
        self,
        *,
        threshold: float = DEFAULT_THRESHOLD,
        models_root: str | Path | None = None,
    ) -> None:
        self.threshold = self._validate_threshold(threshold)
        self.models_root = self._resolve_models_root(models_root)
        self.weights_path = (
            self.models_root / "rfdetr" / MODEL_FILENAME
        ).resolve()

        self._model = None

    @staticmethod
    def _validate_threshold(value: float) -> float:
        value = float(value)

        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"threshold debe estar entre 0 y 1, recibido: {value}"
            )

        return value

    @staticmethod
    def _resolve_models_root(
        explicit_root: str | Path | None,
    ) -> Path:
        if explicit_root is not None:
            return Path(explicit_root).expanduser().resolve()

        env_root = os.getenv("VEO_MATCH_MODELS_ROOT")
        if env_root:
            return Path(env_root).expanduser().resolve()

        # detection_lab/analysis/person/rfdetr_large.py
        # parents[3] -> raíz de VeoMatchLab
        repo_root = Path(__file__).resolve().parents[3]
        return (repo_root / ".model-cache").resolve()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """
        Carga RF-DETR Large desde el checkpoint local.

        No permitimos una descarga silenciosa: si el checkpoint no está
        preparado, fallamos explícitamente.
        """
        if self.loaded:
            return

        if not self.weights_path.is_file():
            raise FileNotFoundError(
                "No se encuentra el checkpoint de RF-DETR Large:\n"
                f"{self.weights_path}\n\n"
                "Prepara los modelos antes de ejecutar el detector."
            )

        # Import lazy para no cargar PyTorch/RF-DETR al importar este módulo.
        from rfdetr import RFDETRLarge

        self._model = RFDETRLarge(
            pretrain_weights=str(self.weights_path)
        )

    def unload(self) -> None:
        """
        Libera el modelo y, si existe CUDA, intenta liberar la caché.
        """
        self._model = None

        gc.collect()

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def detect(
        self,
        image: np.ndarray | Image.Image,
        *,
        threshold: float | None = None,
        color_space: Literal["rgb", "bgr"] = "rgb",
    ) -> list[PersonDetection]:
        """
        Detecta únicamente personas en una imagen.

        Parameters
        ----------
        image:
            PIL Image o ndarray HxWx3.

        threshold:
            Confianza mínima para RF-DETR.
            Si es None se utiliza self.threshold.

        color_space:
            Solo afecta a ndarray:
                "rgb" -> la imagen ya está en RGB.
                "bgr" -> imagen proveniente normalmente de OpenCV.

            PIL siempre se transforma a RGB.

        Returns
        -------
        list[PersonDetection]
            Lista normalizada de personas detectadas.
        """
        if not self.loaded:
            self.load()

        current_threshold = (
            self.threshold
            if threshold is None
            else self._validate_threshold(threshold)
        )

        rgb_image = self._prepare_image(
            image,
            color_space=color_space,
        )

        detections = self._model.predict(
            rgb_image,
            threshold=current_threshold,
        )

        if len(detections) == 0:
            return []

        person_mask = self._person_mask(detections)

        results: list[PersonDetection] = []

        for index in np.flatnonzero(person_mask):
            x1, y1, x2, y2 = detections.xyxy[index]

            confidence = (
                float(detections.confidence[index])
                if detections.confidence is not None
                else 1.0
            )

            results.append(
                PersonDetection(
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                    confidence=confidence,
                )
            )

        # Orden determinista y cómodo para inspección.
        results.sort(
            key=lambda detection: detection.confidence,
            reverse=True,
        )

        return results

    @staticmethod
    def _prepare_image(
        image: np.ndarray | Image.Image,
        *,
        color_space: Literal["rgb", "bgr"],
    ) -> np.ndarray:
        if isinstance(image, Image.Image):
            # V2: usamos np.array(..., copy=True) para obtener memoria writable
            # y evitar el warning de PyTorch de la V1.
            array = np.array(image.convert("RGB"), copy=True)

        elif isinstance(image, np.ndarray):
            array = image

            if array.ndim != 3 or array.shape[2] != 3:
                raise ValueError(
                    "Se esperaba una imagen ndarray HxWx3; "
                    f"shape recibido: {array.shape}"
                )

            if color_space == "bgr":
                # BGR -> RGB sin depender de OpenCV.
                array = array[..., ::-1]

            elif color_space != "rgb":
                raise ValueError(
                    f"color_space no soportado: {color_space}"
                )

            # Garantiza C-contiguous y memoria writable.
            array = np.array(array, copy=True, order="C")

        else:
            raise TypeError(
                "image debe ser numpy.ndarray o PIL.Image.Image"
            )

        return np.ascontiguousarray(array)

    @staticmethod
    def _person_mask(detections) -> np.ndarray:
        """
        Identifica la clase person de forma robusta.

        Preferimos class_name porque evita depender de cómo represente
        internamente RF-DETR los IDs de las clases.

        Dejamos COCO_CLASSES como fallback para el checkpoint COCO oficial.
        """
        class_names = detections.data.get("class_name")

        if class_names is not None:
            names = np.asarray(class_names).astype(str)
            return names == "person"

        if detections.class_id is None:
            raise RuntimeError(
                "RF-DETR no devolvió class_name ni class_id; "
                "no podemos identificar la clase person."
            )

        from rfdetr.assets.coco_classes import COCO_CLASSES

        return np.asarray(
            [
                COCO_CLASSES.get(int(class_id)) == "person"
                for class_id in detections.class_id
            ],
            dtype=bool,
        )


def _to_pil_rgb(
    image: np.ndarray | Image.Image,
    *,
    color_space: Literal["rgb", "bgr"] = "rgb",
) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB").copy()

    if not isinstance(image, np.ndarray):
        raise TypeError("image debe ser numpy.ndarray o PIL.Image.Image")

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            "Se esperaba una imagen ndarray HxWx3; "
            f"shape recibido: {image.shape}"
        )

    array = image

    if color_space == "bgr":
        array = array[..., ::-1]
    elif color_space != "rgb":
        raise ValueError(f"color_space no soportado: {color_space}")

    array = np.array(array, copy=True, order="C")
    return Image.fromarray(array, mode="RGB")


def render_detections(
    image: np.ndarray | Image.Image,
    detections: list[PersonDetection],
    *,
    color_space: Literal["rgb", "bgr"] = "rgb",
    line_width: int = 3,
    show_bottom_center: bool = False,
) -> Image.Image:
    """
    Dibuja bbox + confidence sobre una copia de la imagen original.

    Es solo una utilidad de depuración visual. No altera el algoritmo.
    """
    rendered = _to_pil_rgb(image, color_space=color_space)
    draw = ImageDraw.Draw(rendered)

    try:
        font = ImageFont.load_default(size=18)
    except TypeError:
        # Compatibilidad con versiones de Pillow donde load_default no acepta size.
        font = ImageFont.load_default()

    width, height = rendered.size

    for detection in detections:
        x1 = max(0.0, min(float(width - 1), detection.x1))
        y1 = max(0.0, min(float(height - 1), detection.y1))
        x2 = max(0.0, min(float(width - 1), detection.x2))
        y2 = max(0.0, min(float(height - 1), detection.y2))

        draw.rectangle(
            (x1, y1, x2, y2),
            outline=(255, 0, 0),
            width=max(1, int(line_width)),
        )

        label = f"{detection.confidence:.2f}"

        try:
            text_bbox = draw.textbbox((0, 0), label, font=font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]
        except AttributeError:
            text_w, text_h = draw.textsize(label, font=font)

        pad_x = 4
        pad_y = 2

        label_x = x1
        label_y = max(0.0, y1 - text_h - 2 * pad_y)

        draw.rectangle(
            (
                label_x,
                label_y,
                label_x + text_w + 2 * pad_x,
                label_y + text_h + 2 * pad_y,
            ),
            fill=(255, 0, 0),
        )

        draw.text(
            (label_x + pad_x, label_y + pad_y),
            label,
            fill=(255, 255, 255),
            font=font,
        )

        if show_bottom_center:
            cx, cy = detection.bottom_center
            radius = 4
            draw.ellipse(
                (
                    cx - radius,
                    cy - radius,
                    cx + radius,
                    cy + radius,
                ),
                fill=(255, 255, 0),
            )

    return rendered


def detect_people(
    image: np.ndarray | Image.Image,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    color_space: Literal["rgb", "bgr"] = "rgb",
    models_root: str | Path | None = None,
) -> list[PersonDetection]:
    """
    Atajo funcional para pruebas pequeñas.

    Para vídeo/clips conviene reutilizar una instancia de
    RFDETRLargePersonDetector y no cargar/descargar el modelo por frame.
    """
    detector = RFDETRLargePersonDetector(
        threshold=threshold,
        models_root=models_root,
    )

    try:
        return detector.detect(
            image,
            color_space=color_space,
        )
    finally:
        detector.unload()


def _main() -> None:
    """
    CLI mínimo para validar el algoritmo sin Detection Lab.

    Ejemplo:

        python -m detection_lab.analysis.person.rfdetr_large \
            frame.jpg \
            --threshold 0.25 \
            --output frame_detected.jpg
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Prueba RF-DETR Large sobre una imagen."
    )

    parser.add_argument(
        "image",
        type=Path,
        help="Imagen sobre la que ejecutar detección.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Threshold de detección. Default: {DEFAULT_THRESHOLD}",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Ruta opcional para guardar una copia de la imagen "
            "con bbox y confidence dibujados."
        ),
    )

    parser.add_argument(
        "--show-bottom-center",
        action="store_true",
        help="Dibuja también el bottom-center de cada detección.",
    )

    args = parser.parse_args()

    if not args.image.is_file():
        raise FileNotFoundError(args.image)

    image = Image.open(args.image).convert("RGB")

    detector = RFDETRLargePersonDetector(
        threshold=args.threshold
    )

    try:
        results = detector.detect(image)
    finally:
        detector.unload()

    if args.output is not None:
        rendered = render_detections(
            image,
            results,
            show_bottom_center=args.show_bottom_center,
        )

        args.output.parent.mkdir(parents=True, exist_ok=True)
        rendered.save(args.output, quality=95)

    payload = {
        "model": detector.model_id,
        "image": str(args.image),
        "threshold": args.threshold,
        "persons": len(results),
        "detections": [
            detection.to_dict()
            for detection in results
        ],
    }

    if args.output is not None:
        payload["output"] = str(args.output)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    _main()
