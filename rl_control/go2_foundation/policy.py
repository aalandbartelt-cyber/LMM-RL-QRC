"""Policy backends with a common, dependency-light interface.

Torch and ONNX Runtime are imported lazily.  This lets the contracts and tests
run on a plain Python login machine while the GPU environment supplies the
actual inference dependencies.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol, Sequence, Tuple

from .types import ACTION_DIM, OBSERVATION_DIM


class Policy(Protocol):
    def __call__(self, observation: Sequence[float]) -> Sequence[float]: ...


def validate_policy_io(observation: Sequence[float], action: Sequence[float]) -> Tuple[float, ...]:
    if len(observation) != OBSERVATION_DIM:
        raise ValueError(f"policy expects {OBSERVATION_DIM} observations, got {len(observation)}")
    if len(action) != ACTION_DIM:
        raise ValueError(f"policy must return {ACTION_DIM} actions, got {len(action)}")
    result = tuple(float(v) for v in action)
    # Avoid importing numpy solely for finite checks.
    from math import isfinite

    if not all(isfinite(v) for v in result):
        raise ValueError("policy returned a non-finite action")
    return result


@dataclass
class PolicyRunner:
    """Validate a policy call and retain the last normalized action."""

    policy: Policy
    last_action: Tuple[float, ...] = (0.0,) * ACTION_DIM

    def infer(self, observation: Sequence[float]) -> Tuple[float, ...]:
        action = validate_policy_io(observation, self.policy(observation))
        self.last_action = action
        return action

    def reset(self) -> None:
        self.last_action = (0.0,) * ACTION_DIM


class ZeroPolicy:
    """Deterministic stand-still policy for wiring and safety tests only."""

    def __call__(self, observation: Sequence[float]) -> Tuple[float, ...]:
        if len(observation) != OBSERVATION_DIM:
            raise ValueError(f"expected {OBSERVATION_DIM} observations")
        return (0.0,) * ACTION_DIM


class TorchScriptPolicy:
    """Lazy TorchScript adapter; no hardware actuation is performed here."""

    def __init__(self, model_path: str | Path, device: str = "cpu") -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - depends on deployment image
            raise RuntimeError("TorchScriptPolicy requires PyTorch in the runtime image") from exc
        self._torch = torch
        self._device = torch.device(device)
        self._model = torch.jit.load(str(model_path), map_location=self._device).eval()

    def __call__(self, observation: Sequence[float]) -> Tuple[float, ...]:
        if len(observation) != OBSERVATION_DIM:
            raise ValueError(f"expected {OBSERVATION_DIM} observations")
        with self._torch.inference_mode():
            tensor = self._torch.tensor([list(observation)], dtype=self._torch.float32, device=self._device)
            output = self._model(tensor)
            if hasattr(output, "detach"):
                output = output.detach().cpu().reshape(-1).tolist()
        return validate_policy_io(observation, output)


class OnnxPolicy:
    """Lazy ONNX Runtime adapter for exported policies."""

    def __init__(self, model_path: str | Path, providers: Sequence[str] | None = None) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - depends on deployment image
            raise RuntimeError("OnnxPolicy requires onnxruntime in the runtime image") from exc
        self._session = ort.InferenceSession(str(model_path), providers=list(providers or ["CPUExecutionProvider"]))
        self._input_name = self._session.get_inputs()[0].name

    def __call__(self, observation: Sequence[float]) -> Tuple[float, ...]:
        if len(observation) != OBSERVATION_DIM:
            raise ValueError(f"expected {OBSERVATION_DIM} observations")
        output = self._session.run(None, {self._input_name: [list(observation)]})[0]
        return validate_policy_io(observation, output[0])
