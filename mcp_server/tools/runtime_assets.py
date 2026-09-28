"""Build SDK 4 runtime JSON against parameters read from a genuine Cubism model."""

from __future__ import annotations

import hashlib
import math
import os
import tempfile
from pathlib import Path, PureWindowsPath
from typing import Any


def version_textures(folder: Path, references: dict[str, Any]) -> None:
    """Give texture contents stable cache keys after validating every source and destination.

    Existing files are never overwritten or deleted. Invalid references leave both
    the bundle and references unchanged. Each texture is published only after its
    full contents are written; completed files remain reusable after a failure.
    """
    root = Path(folder).resolve()
    textures = references.get("Textures")
    if not root.is_dir() or not isinstance(textures, list) or not textures:
        raise ValueError("Texture versioning requires a bundle directory and texture list")
    pending = []
    for index, reference in enumerate(textures):
        if not isinstance(reference, str) or not reference or "\x00" in reference:
            raise ValueError("Invalid texture reference")
        relative = Path(reference.replace("\\", "/"))
        if relative.is_absolute() or PureWindowsPath(reference).drive or ".." in relative.parts:
            raise ValueError("Texture path escapes the model bundle")
        source = (root / relative).resolve()
        if not source.is_relative_to(root):
            raise ValueError("Texture path escapes the model bundle")
        data = source.read_bytes()
        digest = hashlib.sha256(data).hexdigest()[:16]
        destination = root / "textures" / f"atlas_{index}_{digest}.png"
        if not destination.resolve().is_relative_to(root):
            raise ValueError("Texture destination escapes the model bundle")
        if destination.parent.exists() and not destination.parent.is_dir():
            raise ValueError("Texture destination parent is not a directory")
        if destination.exists() and (not destination.is_file() or destination.read_bytes() != data):
            raise ValueError("Existing texture cache file has conflicting contents")
        pending.append((destination, data))
    for destination, data in pending:
        destination.parent.mkdir(exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(data)
            os.link(temporary, destination)
        except FileExistsError:
            if not destination.is_file() or destination.read_bytes() != data:
                raise ValueError("Existing texture cache file has conflicting contents") from None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    references["Textures"] = [path.relative_to(root).as_posix() for path, _ in pending]


def _check(parameters: list[dict[str, Any]], parameter_id: str, value: float) -> None:
    parameter = next((p for p in parameters if p["Id"] == parameter_id), None)
    if parameter is None:
        raise ValueError(f"Unknown parameter: {parameter_id}")
    if not math.isfinite(value) or not parameter["Min"] <= value <= parameter["Max"]:
        raise ValueError(f"Parameter value outside authored range: {parameter_id}")


def expression(
    parameters: list[dict[str, Any]],
    values: dict[str, float],
    *,
    multiply: set[str] | None = None,
) -> dict[str, Any]:
    """Build a fading expression, allowing eye multipliers to retain motion-driven blinks."""
    multiply = multiply or set()
    if multiply - values.keys():
        raise ValueError("Unknown expression multiplier")
    for key, value in values.items():
        _check(parameters, key, value)
    return {
        "Type": "Live2D Expression",
        "FadeInTime": 0.35,
        "FadeOutTime": 0.45,
        "Parameters": [
            {"Id": key, "Value": value, "Blend": "Multiply" if key in multiply else "Overwrite"}
            for key, value in values.items()
        ],
    }


def motion(
    parameters: list[dict[str, Any]],
    duration: float,
    curves: dict[str, list[tuple[float, float]]],
    *,
    loop: bool = False,
) -> dict[str, Any]:
    """Build linear sampled curves with exact SDK allocation counts and loop continuity."""
    if not math.isfinite(duration) or duration <= 0 or not curves:
        raise ValueError("Motion requires positive duration and curves")
    result = []
    points = segments = 0
    for key, samples in curves.items():
        if len(samples) < 2 or samples[0][0] != 0 or samples[-1][0] != duration:
            raise ValueError("Curve time must span the entire motion")
        previous = -1.0
        encoded: list[float | int] = []
        for index, (timestamp, value) in enumerate(samples):
            if not math.isfinite(timestamp) or timestamp <= previous or timestamp > duration:
                raise ValueError("Curve time must strictly increase")
            _check(parameters, key, value)
            if index:
                encoded.append(0)
            encoded.extend([timestamp, value])
            previous = timestamp
        if loop and samples[0][1] != samples[-1][1]:
            raise ValueError("A loop must return to its initial value")
        result.append({"Target": "Parameter", "Id": key, "Segments": encoded})
        points += len(samples)
        segments += len(samples) - 1
    return {
        "Version": 3,
        "Meta": {
            "Duration": duration,
            "Fps": 30,
            "Loop": loop,
            "AreBeziersRestricted": True,
            "CurveCount": len(result),
            "TotalSegmentCount": segments,
            "TotalPointCount": points,
            "UserDataCount": 0,
            "TotalUserDataSize": 0,
            "FadeInTime": 0.5,
            "FadeOutTime": 0.5,
        },
        "Curves": result,
        "UserData": [],
    }


def physics_group(
    parameters: list[dict[str, Any]], group_id: str, source: str, output: str
) -> dict[str, Any]:
    """Create a damped two-particle cloth group without parameter feedback."""
    for key in (source, output):
        parameter = next((p for p in parameters if p["Id"] == key), None)
        _check(parameters, key, float(parameter["Default"]) if parameter else 0)
    if source == output:
        raise ValueError("Physics parameter feedback is not allowed")
    return {
        "Id": group_id,
        "Input": [
            {
                "Source": {"Target": "Parameter", "Id": source},
                "Weight": 100,
                "Type": "X",
                "Reflect": False,
            }
        ],
        "Output": [
            {
                "Destination": {"Target": "Parameter", "Id": output},
                "VertexIndex": 1,
                "Scale": 1.5,
                "Weight": 100,
                "Type": "Angle",
                "Reflect": False,
            }
        ],
        "Vertices": [
            {
                "Position": {"X": 0, "Y": 0},
                "Mobility": 1,
                "Delay": 1,
                "Acceleration": 1,
                "Radius": 0,
            },
            {
                "Position": {"X": 0, "Y": 8},
                "Mobility": 0.82,
                "Delay": 0.28,
                "Acceleration": 1.1,
                "Radius": 8,
            },
        ],
        "Normalization": {
            key: {"Minimum": -10, "Default": 0, "Maximum": 10} for key in ("Position", "Angle")
        },
    }
