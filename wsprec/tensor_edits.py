"""Compile declarative numerical edits against an independent copy of a tensor state."""

import io
import math
from pathlib import Path

import numpy as np
import torch


_MAX_ASSET_BYTES = 20 * 1024 * 1024
_OPS = {"add", "multiply", "replace", "left_multiply", "right_multiply"}


def _asset_array(directory, name):
    if type(name) is not str:
        raise ValueError("asset must be a relative .npy path")
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".npy":
        raise ValueError("asset must be a relative .npy path without parent references")
    try:
        root = Path(directory).resolve(strict=True)
        path = (root / relative).resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("asset must be a file inside the asset directory")
        if path.stat().st_size > _MAX_ASSET_BYTES:
            raise ValueError("asset exceeds the size limit")
        with path.open("rb") as stream:
            data = stream.read(_MAX_ASSET_BYTES + 1)
        if len(data) > _MAX_ASSET_BYTES:
            raise ValueError("asset exceeds the size limit")
        # Check the header and byte count before allocating from untrusted dimensions.
        stream = io.BytesIO(data)
        version = np.lib.format.read_magic(stream)
        readers = {(1, 0): np.lib.format.read_array_header_1_0,
                   (2, 0): np.lib.format.read_array_header_2_0,
                   (3, 0): np.lib.format.read_array_header_2_0}
        if version not in readers:
            raise ValueError("unsupported .npy version")
        shape, fortran, dtype = readers[version](stream)
        if dtype.kind not in "iuf" or dtype.itemsize * math.prod(shape) != len(data) - stream.tell():
            raise ValueError("invalid numeric .npy payload")
        return np.frombuffer(data, dtype=dtype, offset=stream.tell()).reshape(
            shape, order="F" if fortran else "C")
    except (OSError, EOFError, RuntimeError, OverflowError) as exc:
        raise ValueError("invalid asset") from exc


def _check_literal(value):
    if type(value) is list:
        for child in value:
            _check_literal(child)
    elif type(value) not in (int, float):
        raise ValueError("value must consist of real numbers")


def _operand(operation, target, directory):
    if "value" in operation:
        _check_literal(operation["value"])
        array = np.asarray(operation["value"])
    else:
        array = _asset_array(directory, operation["asset"])
    if array.dtype.kind not in "iuf" or not np.isfinite(array).all():
        raise ValueError("operand must contain only finite real numbers")
    # Materialize in native byte order for Torch, including non-native .npy files.
    operand = torch.as_tensor(array.astype(np.float64), device=target.device).to(target.dtype)
    if not torch.isfinite(operand).all():
        raise ValueError("operand overflows the target dtype")
    return operand


def _apply(target, op, operand):
    if op in {"add", "multiply"}:
        try:
            broadcast = torch.broadcast_shapes(target.shape, operand.shape)
        except RuntimeError as exc:
            raise ValueError("operand cannot broadcast to target") from exc
        if broadcast != target.shape:
            raise ValueError("operation would change parameter shape")
        return target + operand if op == "add" else target * operand
    if op == "replace":
        if operand.ndim != 0 and operand.shape != target.shape:
            raise ValueError("replacement must be scalar or match the target shape")
        return operand.expand_as(target).clone()
    if target.ndim != 2 or operand.ndim != 2:
        raise ValueError("matrix multiplication requires two matrices")
    side = target.shape[0 if op == "left_multiply" else 1]
    if operand.shape != (side, side):
        raise ValueError("matrix operand must preserve the parameter shape")
    return operand @ target if op == "left_multiply" else target @ operand


@torch.no_grad()
def compile_patch(base_state: dict[str, torch.Tensor], patch: dict, asset_directory: Path):
    """Return a detached edited state and scalar-element change statistics."""
    if type(patch) is not dict or set(patch) != {"id", "operations"} or type(patch["id"]) is not str:
        raise ValueError("patch requires an id and operations")
    operations = patch["operations"]
    if type(operations) is not list or len(operations) > 6:
        raise ValueError("operations must be a list of at most six entries")
    if any(value.dtype == torch.bool or value.is_complex() for value in base_state.values()):
        raise ValueError("base parameters must be real numerical tensors")
    edited = {key: value.detach().clone() for key, value in base_state.items()}
    for op in operations:
        if type(op) is not dict or set(op) not in (
            {"parameter", "op", "value"}, {"parameter", "op", "asset"}
        ):
            raise ValueError("operation requires a parameter, op and exactly one operand")
        name, kind = op["parameter"], op["op"]
        if type(name) is not str or name not in edited or type(kind) is not str or kind not in _OPS:
            raise ValueError("unknown parameter or operation")
        target = edited[name]
        operand = _operand(op, target, asset_directory)
        edited[name] = _apply(target, kind, operand)
        if not torch.isfinite(edited[name]).all():
            raise ValueError("operation produced a non-finite parameter")

    changed = total = 0
    base_norm = delta_norm = 0.0
    for key, result in edited.items():
        source = base_state[key].detach().to(device="cpu", dtype=torch.float64)
        value = result.to(device="cpu", dtype=torch.float64)
        if not torch.isfinite(source).all() or not torch.isfinite(value).all() or (value.abs() > 100).any():
            raise ValueError("state exceeds finite-value or magnitude limits")
        delta = value - source
        changed += int(torch.count_nonzero(delta))
        total += value.numel()
        delta_norm = math.hypot(delta_norm, float(torch.linalg.vector_norm(delta)))
        base_norm = math.hypot(base_norm, float(torch.linalg.vector_norm(source)))
    relative = delta_norm / base_norm if base_norm else (math.inf if delta_norm else 0.0)
    if relative > 0.5 + 1e-7:
        raise ValueError("patch exceeds the relative L2 change budget")
    return edited, {"changed_parameters": changed, "total_parameters": total,
                    "delta_l2": delta_norm, "relative_delta_l2": relative}
