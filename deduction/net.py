"""Small multi-label network saved as a NumPy archive.

The network estimates functional-group probabilities from the same features
the rule engine uses. Structure proposals still come from the rules, which
can show the evidence for each call.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from deduction.features import LABELS


def _relu(values: np.ndarray) -> np.ndarray:
    return np.maximum(values, 0.0)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -40, 40)
    return 1.0 / (1.0 + np.exp(-clipped))


class FunctionalGroupNet:
    def __init__(self, sizes: list[int], seed: int = 7):
        rng = np.random.default_rng(seed)
        self.sizes = sizes
        self.weights = []
        self.biases = []
        for fan_in, fan_out in zip(sizes, sizes[1:]):
            scale = np.sqrt(2.0 / fan_in)
            self.weights.append(rng.normal(0, scale, size=(fan_in, fan_out)).astype(np.float32))
            self.biases.append(np.zeros(fan_out, dtype=np.float32))

    def forward(self, features: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
        activations = [features.astype(np.float32)]
        current = activations[0]
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            current = current @ weight + bias
            if index < len(self.weights) - 1:
                current = _relu(current)
            else:
                current = _sigmoid(current)
            activations.append(current)
        return activations, current

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        row = np.atleast_2d(features).astype(np.float32)
        _activations, output = self.forward(row)
        return output[0]

    def fit(self, features: np.ndarray, labels: np.ndarray, epochs: int = 80, learning_rate: float = 0.05, seed: int = 7) -> list[float]:
        rng = np.random.default_rng(seed)
        count = features.shape[0]
        losses = []
        for _epoch in range(epochs):
            order = rng.permutation(count)
            total = 0.0
            for start in range(0, count, 32):
                batch = order[start : start + 32]
                loss, gradients_w, gradients_b = self._batch_gradients(features[batch], labels[batch])
                total += loss * len(batch)
                for index in range(len(self.weights)):
                    self.weights[index] -= learning_rate * gradients_w[index]
                    self.biases[index] -= learning_rate * gradients_b[index]
            losses.append(total / count)
        return losses

    def _batch_gradients(self, features: np.ndarray, labels: np.ndarray):
        activations, output = self.forward(features)
        batch = max(features.shape[0], 1)
        # Binary cross-entropy derivative at the sigmoid output.
        delta = (output - labels) / batch
        loss = float(
            -np.mean(labels * np.log(output + 1e-7) + (1 - labels) * np.log(1 - output + 1e-7))
        )
        grad_w = []
        grad_b = []
        for index in reversed(range(len(self.weights))):
            grad_w.append(activations[index].T @ delta)
            grad_b.append(delta.sum(axis=0))
            if index == 0:
                break
            delta = (delta @ self.weights[index].T) * (activations[index] > 0)
        grad_w.reverse()
        grad_b.reverse()
        return loss, grad_w, grad_b

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"sizes": np.asarray(self.sizes, dtype=np.int32), "labels": np.asarray(LABELS)}
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            payload[f"w{index}"] = weight
            payload[f"b{index}"] = bias
        np.savez(path, **payload)

    @classmethod
    def load(cls, path: str | Path) -> "FunctionalGroupNet":
        archive = np.load(path, allow_pickle=False)
        sizes = [int(value) for value in archive["sizes"]]
        model = cls(sizes)
        for index in range(len(sizes) - 1):
            model.weights[index] = archive[f"w{index}"]
            model.biases[index] = archive[f"b{index}"]
        return model

    def probabilities(self, features: np.ndarray) -> dict[str, float]:
        values = self.predict_proba(features)
        return {label: round(float(value), 3) for label, value in zip(LABELS, values)}
