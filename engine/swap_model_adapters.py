from __future__ import annotations

import cv2
import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper


_ARCFACE_POINTS = np.array([
    [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
    [41.5493, 92.3655], [70.7299, 92.2041],
], dtype=np.float32)


class ReswapperModel:
    """ONNX adapter for ReSwapper's INSwapper-compatible 128/256 models."""

    def __init__(self, model_path: str, providers):
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_names = [item.name for item in self.session.get_inputs()]
        self.output_names = [item.name for item in self.session.get_outputs()]
        shape = self.session.get_inputs()[0].shape
        self.input_size = (int(shape[3]), int(shape[2]))
        self.input_mean = 0.0
        self.input_std = 255.0
        graph = onnx.load(model_path, load_external_data=False)
        self.emap = numpy_helper.to_array(graph.graph.initializer[-1])

    def get(self, image, target_face, source_face, paste_back=False):
        ratio = float(self.input_size[0]) / 128.0
        points = _ARCFACE_POINTS.copy() * ratio
        points[:, 0] += 8.0 * ratio
        matrix, _ = cv2.estimateAffinePartial2D(
            np.asarray(target_face.kps, dtype=np.float32), points
        )
        aligned = cv2.warpAffine(image, matrix, self.input_size, borderValue=0.0)
        blob = cv2.dnn.blobFromImage(
            aligned, 1.0 / self.input_std, self.input_size,
            (self.input_mean,) * 3, swapRB=True,
        )
        embedding = np.asarray(source_face.normed_embedding, dtype=np.float32).reshape(1, -1)
        latent = embedding @ self.emap
        latent /= max(float(np.linalg.norm(latent)), 1e-8)
        prediction = self.session.run(
            self.output_names,
            {self.input_names[0]: blob, self.input_names[1]: latent.astype(np.float32)},
        )[0]
        patch = prediction.transpose((0, 2, 3, 1))[0]
        patch = np.clip(255.0 * patch, 0, 255).astype(np.uint8)[:, :, ::-1]
        return (patch, matrix) if not paste_back else patch


class HyperSwapModel:
    """ONNX adapter for FaceFusion HyperSwap 256 models."""

    input_size = (256, 256)

    def __init__(self, model_path: str, providers):
        self.session = ort.InferenceSession(model_path, providers=providers)

    def get(self, image, target_face, source_face, paste_back=False):
        target_points = np.asarray(target_face.kps, dtype=np.float32)
        reference = np.array([
            [84.87, 105.94], [171.13, 105.94], [128.0, 146.66],
            [96.95, 188.64], [159.05, 188.64],
        ], dtype=np.float32)
        matrix, _ = cv2.estimateAffinePartial2D(target_points, reference)
        crop = cv2.warpAffine(
            image, matrix, self.input_size,
            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT,
        )
        target = crop[:, :, ::-1].astype(np.float32) / 127.5 - 1.0
        target = target.transpose(2, 0, 1)[None]
        source = np.asarray(source_face.normed_embedding, dtype=np.float32).reshape(1, -1)
        output = self.session.run(None, {"source": source, "target": target})[0][0]
        output = np.nan_to_num(output, nan=0.0, posinf=1.0, neginf=-1.0)
        if output.min() < 0.0 or output.max() <= 1.5:
            output = (output + 1.0) * 127.5
        patch = np.clip(output, 0, 255).astype(np.uint8).transpose(1, 2, 0)[:, :, ::-1]
        return (patch, matrix) if not paste_back else patch
