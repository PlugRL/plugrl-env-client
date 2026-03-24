from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from plugrl_env_client.envs.base_env import Observation


def to_builtin(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(v) for v in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(to_builtin(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(to_builtin(payload), ensure_ascii=False) + "\n")


def stats_for_array(arr: np.ndarray) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
    }
    if arr.size == 0:
        return stats
    if np.issubdtype(arr.dtype, np.number) or np.issubdtype(arr.dtype, np.bool_):
        numeric = arr.astype(np.float32, copy=False)
        stats.update(
            {
                "mean": float(np.mean(numeric)),
                "std": float(np.std(numeric)),
                "min": float(np.min(numeric)),
                "max": float(np.max(numeric)),
            }
        )
    return stats


def stats_for_text(text_arr: np.ndarray) -> dict[str, Any]:
    values = np.asarray(text_arr, dtype=np.str_)
    return {
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "value": values.tolist() if values.ndim else str(values.item()),
    }


def ensure_rgb(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return np.repeat(frame[..., None], 3, axis=2)
    if frame.ndim == 3 and frame.shape[-1] == 1:
        return np.repeat(frame, 3, axis=2)
    if frame.ndim == 3 and frame.shape[-1] >= 3:
        return frame[..., :3]
    raise ValueError(f"Unsupported image shape for visualization: {frame.shape}")


def make_grid(frames: np.ndarray) -> np.ndarray:
    if frames.ndim != 4:
        raise ValueError(
            f"Expected image batch with shape (b, h, w, c), got {frames.shape}"
        )
    batch, height, width, channels = frames.shape
    cols = int(np.ceil(np.sqrt(batch)))
    rows = int(np.ceil(batch / cols))
    grid = np.zeros((rows * height, cols * width, channels), dtype=frames.dtype)
    for idx in range(batch):
        row = idx // cols
        col = idx % cols
        grid[
            row * height : (row + 1) * height,
            col * width : (col + 1) * width,
            :,
        ] = frames[idx]
    return grid


def phase_stats(obs: Observation) -> dict[str, Any]:
    return {
        "images": {key: stats_for_array(value[0]) for key, value in obs.images.items()},
        "states": {key: stats_for_array(value[0]) for key, value in obs.states.items()},
        "text": stats_for_text(obs.text),
    }
