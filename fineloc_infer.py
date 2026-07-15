#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import open3d as o3d
except ImportError as exc:
    raise SystemExit("open3d is required. Install open3d or use the packaged executable.") from exc


STAGE1_POINTS = 8192
STAGE2_KNN_POINTS = 16384


def resource_path(relative: str) -> Path:
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    candidate = base / relative
    if candidate.exists():
        return candidate
    return Path(__file__).resolve().parent / relative


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        return json.load(f)


def read_pcd_xyz(path: Path) -> np.ndarray:
    """Read binary xyz PCD files used by the FineLocation datasets."""
    with path.open("rb") as f:
        points = None
        fields = sizes = types = None
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"PCD DATA line not found: {path}")
            text = line.decode("latin1").strip()
            parts = text.split()
            if not parts:
                continue
            key = parts[0]
            if key == "FIELDS":
                fields = parts[1:]
            elif key == "SIZE":
                sizes = [int(x) for x in parts[1:]]
            elif key == "TYPE":
                types = parts[1:]
            elif key == "POINTS":
                points = int(parts[1])
            elif key == "DATA":
                data_mode = parts[1]
                break

        if points is None or fields is None or sizes is None or types is None:
            raise ValueError(f"incomplete PCD header: {path}")
        if (
            data_mode != "binary"
            or fields[:3] != ["x", "y", "z"]
            or sizes[:3] != [4, 4, 4]
            or types[:3] != ["F", "F", "F"]
        ):
            raise ValueError(f"unsupported PCD format: {path}")
        raw = f.read(points * 12)

    xyz = np.frombuffer(raw, dtype=np.float32).reshape(points, 3).copy()
    return xyz[np.isfinite(xyz).all(axis=1)]


def normalize_np(vector: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < eps:
        return np.zeros_like(vector, dtype=np.float32)
    return (vector / norm).astype(np.float32)


def line_frame_from_end_delta(end_delta: np.ndarray) -> np.ndarray:
    x_axis = normalize_np(np.asarray(end_delta, dtype=np.float32).reshape(3))
    if not np.any(x_axis):
        return np.eye(3, dtype=np.float32)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if abs(float(np.dot(up, x_axis))) > 0.95:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    y_axis = normalize_np(np.cross(up, x_axis))
    if not np.any(y_axis):
        y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    z_axis = normalize_np(np.cross(x_axis, y_axis))
    return np.stack([x_axis, y_axis, z_axis], axis=1).astype(np.float32)


def select_plane_normals(param: dict[str, Any], end_delta: np.ndarray) -> tuple[np.ndarray, tuple[int, int], float]:
    normals: dict[int, np.ndarray] = {}
    for index in (1, 2, 3):
        normal = normalize_np(np.asarray(param.get(f"normalPlane{index}", [0, 0, 0]), dtype=np.float32))
        if np.any(normal):
            normals[index] = normal

    seam_dir = normalize_np(end_delta)
    best_pair: tuple[int, int] | None = None
    best_angle = float("inf")
    for first, second in ((1, 2), (1, 3), (2, 3)):
        if first not in normals or second not in normals:
            continue
        intersection = normalize_np(np.cross(normals[first], normals[second]))
        if not np.any(intersection) or not np.any(seam_dir):
            continue
        cosine = float(np.clip(abs(np.dot(intersection, seam_dir)), 0.0, 1.0))
        angle = math.degrees(math.acos(cosine))
        if angle < best_angle:
            best_pair = (first, second)
            best_angle = angle

    if best_pair is None:
        raise ValueError("param does not contain two valid normalPlane vectors")
    selected = np.stack([normals[best_pair[0]], normals[best_pair[1]]], axis=0).astype(np.float32)
    return selected, best_pair, best_angle


def plane_basis_frame(end_delta: np.ndarray, plane_normals: np.ndarray) -> tuple[np.ndarray, float, bool]:
    """Build p_basis = p_ref @ frame from seam direction and two plane normals."""
    x_axis = normalize_np(end_delta)
    normal1 = normalize_np(np.asarray(plane_normals[0], dtype=np.float32))
    normal2 = normalize_np(np.asarray(plane_normals[1], dtype=np.float32))
    basis = np.stack([x_axis, normal1, normal2], axis=1).astype(np.float32)
    determinant = float(np.linalg.det(basis))
    if not np.isfinite(basis).all() or abs(determinant) < 1e-4:
        return line_frame_from_end_delta(end_delta), determinant, True
    return np.linalg.inv(basis.T).astype(np.float32), determinant, False


def vector_from_basis(vector_local: np.ndarray, frame: np.ndarray) -> np.ndarray:
    return (np.asarray(vector_local, dtype=np.float32) @ np.linalg.inv(frame)).astype(np.float32)


def normal_from_basis(normal_local: np.ndarray, frame: np.ndarray) -> np.ndarray:
    # Training transforms plane normals with inverse-transpose. This is its inverse.
    return normalize_np(np.asarray(normal_local, dtype=np.float32) @ frame.T)


def fps_downsample(points: np.ndarray, num_points: int) -> np.ndarray:
    if points.shape[0] < num_points:
        raise ValueError(f"point count {points.shape[0]} < requested {num_points}")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64, copy=False))
    sampled = pcd.farthest_point_down_sample(num_points, start_index=0)
    out = np.asarray(sampled.points, dtype=np.float32)
    if out.shape[0] != num_points:
        raise RuntimeError(f"FPS returned {out.shape[0]} points, expected {num_points}")
    return out


def random_downsample(points: np.ndarray, num_points: int, seed: int) -> np.ndarray:
    if points.shape[0] < num_points:
        raise ValueError(f"point count {points.shape[0]} < requested {num_points}")
    rng = np.random.default_rng(seed)
    indices = rng.choice(points.shape[0], size=num_points, replace=False)
    return points[indices].astype(np.float32, copy=False)


def crop_knn_around_delta(
    raw_points: np.ndarray,
    ref_point: np.ndarray,
    center_delta: np.ndarray,
    num_points: int,
) -> tuple[np.ndarray, float]:
    raw_local = raw_points.astype(np.float32, copy=False) - ref_point[None, :]
    diff = raw_local - center_delta[None, :]
    dist2 = np.sum(diff * diff, axis=1)
    k = min(int(num_points), raw_local.shape[0])
    idx = np.argpartition(dist2, kth=k - 1)[:k]
    crop = raw_local[idx]
    radius = float(np.sqrt(dist2[idx].max()))
    if crop.shape[0] < num_points:
        rng = np.random.default_rng(0)
        extra = rng.choice(crop.shape[0], size=num_points - crop.shape[0], replace=True)
        crop = np.concatenate([crop, crop[extra]], axis=0)
    return (crop - center_delta[None, :]).astype(np.float32), radius


class PointHeatmapOffsetNet(nn.Module):
    """Point-wise heatmap + offset model used for both stages."""

    def __init__(
        self,
        dropout: float = 0.0,
        include_line_head: bool = False,
        include_plane_head: bool = False,
    ) -> None:
        super().__init__()
        self.dropout = float(dropout)
        self.include_line_head = bool(include_line_head)
        self.include_plane_head = bool(include_plane_head)
        self.point_encoder = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )
        self.point_head = nn.Sequential(
            nn.Conv1d(256 + 256 + 3, 256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout) if self.dropout > 0 else nn.Identity(),
            nn.Conv1d(256, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        self.heatmap_head = nn.Conv1d(128, 1, kernel_size=1)
        self.offset_head = nn.Conv1d(128, 3, kernel_size=1)
        if self.include_line_head:
            self.line_head = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(p=self.dropout) if self.dropout > 0 else nn.Identity(),
                nn.Linear(128, 3),
            )
        if self.include_plane_head:
            self.plane_head = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(p=self.dropout) if self.dropout > 0 else nn.Identity(),
                nn.Linear(128, 6),
            )

    def forward(self, points: torch.Tensor, _params: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        xyz = points.transpose(1, 2)
        point_feat = self.point_encoder(xyz)
        global_vec = torch.max(point_feat, dim=2)[0]
        global_feat = global_vec.unsqueeze(-1).expand(-1, -1, points.shape[1])
        point_context = torch.cat([point_feat, global_feat, xyz], dim=1)
        point_context = self.point_head(point_context)
        heatmap_logits = self.heatmap_head(point_context).squeeze(1)
        offsets = self.offset_head(point_context).transpose(1, 2)
        candidate_points = points + offsets
        weights = torch.softmax(heatmap_logits, dim=1)
        pred_delta = torch.sum(candidate_points * weights.unsqueeze(-1), dim=1)
        outputs = {
            "pred_delta": pred_delta,
            "heatmap_logits": heatmap_logits,
            "offsets": offsets,
            "weights": weights,
        }
        if self.include_line_head:
            outputs["pred_line_dir"] = F.normalize(self.line_head(global_vec), dim=1, eps=1e-6)
        if self.include_plane_head:
            outputs["pred_plane_normals"] = F.normalize(
                self.plane_head(global_vec).view(-1, 2, 3), dim=2, eps=1e-6
            )
        return outputs


def load_model(checkpoint_path: Path, device: torch.device) -> PointHeatmapOffsetNet:
    if sys.platform.startswith("win"):
        # The checkpoints were saved on Linux and contain pathlib.PosixPath
        # objects in metadata. Windows cannot unpickle PosixPath directly.
        original_posix_path = pathlib.PosixPath
        try:
            pathlib.PosixPath = pathlib.WindowsPath
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        finally:
            pathlib.PosixPath = original_posix_path
    else:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = checkpoint["model_state"]
    has_line_head = any(key.startswith("line_head.") for key in state)
    has_plane_head = any(key.startswith("plane_head.") for key in state)
    model = PointHeatmapOffsetNet(
        include_line_head=has_line_head,
        include_plane_head=has_plane_head,
    ).to(device)
    result = model.load_state_dict(state, strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"incompatible checkpoint: missing={result.missing_keys}, unexpected={result.unexpected_keys}")
    model.eval()
    return model


@torch.no_grad()
def predict_delta_global(
    model: nn.Module,
    points_global_ref_frame: np.ndarray,
    frame: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    points_basis = points_global_ref_frame.astype(np.float32) @ frame
    points = torch.from_numpy(points_basis).unsqueeze(0).to(device)
    params = torch.zeros((1, 0), dtype=torch.float32, device=device)
    outputs = model(points, params)
    pred_local = outputs["pred_delta"].squeeze(0).detach().cpu().numpy().astype(np.float32)
    pred_delta = vector_from_basis(pred_local, frame)
    pred_line_global = None
    if "pred_line_dir" in outputs:
        pred_line_local = outputs["pred_line_dir"].squeeze(0).detach().cpu().numpy().astype(np.float32)
        pred_line_global = normalize_np(vector_from_basis(pred_line_local, frame))
    pred_planes_global = None
    if "pred_plane_normals" in outputs:
        pred_planes_local = outputs["pred_plane_normals"].squeeze(0).detach().cpu().numpy().astype(np.float32)
        pred_planes_global = np.stack(
            [normal_from_basis(pred_planes_local[index], frame) for index in range(2)], axis=0
        ).astype(np.float32)
    return pred_delta, pred_line_global, pred_planes_global


def is_valid_point(point: np.ndarray, eps: float = 1e-6) -> bool:
    return bool(np.isfinite(point).all() and np.linalg.norm(point.astype(np.float32)) > eps)


def decompose_error(err: np.ndarray, end_delta: np.ndarray) -> dict[str, float]:
    norm = float(np.linalg.norm(end_delta))
    if norm < 1e-6:
        return {"parallel_signed_mm": 0.0, "parallel_abs_mm": 0.0, "perp_mm": float(np.linalg.norm(err))}
    direction = end_delta.astype(np.float32) / norm
    parallel_signed = float(np.dot(err, direction))
    parallel_vec = parallel_signed * direction
    return {
        "parallel_signed_mm": parallel_signed,
        "parallel_abs_mm": abs(parallel_signed),
        "perp_mm": float(np.linalg.norm(err - parallel_vec)),
    }


def make_sphere(center: np.ndarray, radius: float, color: list[float]) -> o3d.geometry.TriangleMesh:
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(color)
    mesh.translate(center.astype(np.float64))
    return mesh


def make_line(points: np.ndarray, lines: list[list[int]], colors: list[list[float]]) -> o3d.geometry.LineSet:
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    line_set.lines = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
    line_set.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
    return line_set


def visualize_prediction(
    raw_points: np.ndarray,
    ref_point: np.ndarray,
    end_delta: np.ndarray,
    coarse_delta: np.ndarray,
    final_delta: np.ndarray,
    final_line_dir: np.ndarray | None,
    tool_point: np.ndarray | None,
    tool_line_dir: np.ndarray | None,
    point_size: float,
    marker_radius: float,
) -> None:
    raw_local = raw_points.astype(np.float32, copy=False) - ref_point[None, :]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(raw_local.astype(np.float64))
    pcd.paint_uniform_color([0.25, 0.25, 0.25])

    ref = np.zeros(3, dtype=np.float32)
    pred = final_delta.astype(np.float32)
    coarse = coarse_delta.astype(np.float32)
    end = end_delta.astype(np.float32)
    points = [ref, pred, coarse, end]
    lines = [[0, 3]]
    colors = [[0.55, 0.2, 0.85]]
    geoms: list[o3d.geometry.Geometry] = [
        pcd,
        make_sphere(ref, marker_radius, [0.1, 0.3, 0.95]),
        make_sphere(pred, marker_radius, [0.0, 0.72, 0.22]),
        make_sphere(coarse, marker_radius * 0.8, [1.0, 0.8, 0.05]),
        make_sphere(end, marker_radius * 0.75, [0.55, 0.2, 0.85]),
    ]
    if tool_point is not None and is_valid_point(tool_point):
        tool = tool_point.astype(np.float32) - ref_point
        geoms.append(make_sphere(tool, marker_radius, [0.95, 0.12, 0.12]))
        if tool_line_dir is not None and np.linalg.norm(tool_line_dir) > 1e-6:
            tool_idx = len(points)
            points.append(tool)
            tool_dir = normalize_np(tool_line_dir.astype(np.float32))
            tool_line_end = tool + tool_dir * float(np.linalg.norm(end_delta))
            tool_line_idx = len(points)
            points.append(tool_line_end)
            lines.append([tool_idx, tool_line_idx])
            colors.append([1.0, 0.45, 0.0])
    if final_line_dir is not None and np.linalg.norm(final_line_dir) > 1e-6:
        line_dir = normalize_np(final_line_dir.astype(np.float32))
        line_end = pred + line_dir * float(np.linalg.norm(end_delta))
        line_idx = len(points)
        points.append(line_end)
        lines.append([1, line_idx])
        colors.append([0.0, 0.75, 0.9])
    geoms.insert(1, make_line(np.stack(points, axis=0), lines, colors))

    print("colors: gray cloud, blue reference, red tool output point, orange tool line direction, green network output point, yellow stage1 point, purple seam prior, cyan network line direction")
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="FineLocation inference", width=1280, height=860)
    for geom in geoms:
        vis.add_geometry(geom)
    opt = vis.get_render_option()
    opt.point_size = float(point_size)
    opt.background_color = np.asarray([1.0, 1.0, 1.0], dtype=np.float64)
    vis.run()
    vis.destroy_window()


def find_matching_files(raw_dir: Path, recursive: bool) -> list[tuple[Path, Path, Path | None]]:
    pattern = "**/result_*.txt" if recursive else "result_*.txt"
    triples: list[tuple[Path, Path, Path | None]] = []
    for result_path in sorted(raw_dir.glob(pattern)):
        sample_dir = result_path.parent
        suffix = result_path.name[len("result_"):]
        stem = suffix[:-4] if suffix.endswith(".txt") else suffix
        cloud = sample_dir / f"cloud_{stem}.pcd"
        param = sample_dir / f"param{stem}.txt"
        if cloud.exists() and param.exists():
            triples.append((cloud, param, result_path))
    return triples


def infer_one(
    cloud_path: Path,
    param_path: Path,
    result_path: Path | None,
    stage1_model: nn.Module,
    stage2_model: nn.Module,
    device: torch.device,
    stage1_points: int,
    stage2_knn_points: int,
    stage1_sampling: str,
    sampling_seed: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    start = time.perf_counter()
    param = load_json(param_path)
    ref_point = np.asarray(param["startPos"], dtype=np.float32)
    end_delta = np.asarray(param["endPos1"], dtype=np.float32) - ref_point
    selected_normals, selected_indices, pair_angle_deg = select_plane_normals(param, end_delta)
    frame, basis_determinant, basis_fallback = plane_basis_frame(end_delta, selected_normals)
    raw_points = read_pcd_xyz(cloud_path)

    if stage1_sampling == "random":
        sampled_cloud = random_downsample(raw_points, stage1_points, sampling_seed)
    else:
        sampled_cloud = fps_downsample(raw_points, stage1_points)
    stage1_cloud = sampled_cloud - ref_point[None, :]
    coarse_delta, coarse_line_dir, coarse_plane_normals = predict_delta_global(
        stage1_model, stage1_cloud, frame, device
    )
    refine_points, radius = crop_knn_around_delta(raw_points, ref_point, coarse_delta, stage2_knn_points)
    residual_delta, final_line_dir, final_plane_normals = predict_delta_global(
        stage2_model, refine_points, frame, device
    )
    final_delta = coarse_delta + residual_delta
    final_point = ref_point + final_delta

    row: dict[str, Any] = {
        "cloud": str(cloud_path.resolve()),
        "param": str(param_path.resolve()),
        "result": "" if result_path is None else str(result_path.resolve()),
        "raw_point_count": int(raw_points.shape[0]),
        "stage1_sampling": stage1_sampling,
        "sampling_seed": int(sampling_seed),
        "stage1_points": int(stage1_points),
        "stage2_knn_points": int(stage2_knn_points),
        "stage2_knn_radius_mm": float(radius),
        "ref_point": ref_point.tolist(),
        "end_delta": end_delta.tolist(),
        "selected_param_plane_indices": list(selected_indices),
        "selected_param_plane_normals": selected_normals.tolist(),
        "selected_plane_intersection_angle_deg": float(pair_angle_deg),
        "plane_basis_determinant": float(basis_determinant),
        "plane_basis_fallback_to_line_frame": bool(basis_fallback),
        "coarse_delta": coarse_delta.tolist(),
        "coarse_point": (ref_point + coarse_delta).tolist(),
        "coarse_line_dir": None if coarse_line_dir is None else coarse_line_dir.tolist(),
        "coarse_plane_normals": None if coarse_plane_normals is None else coarse_plane_normals.tolist(),
        "residual_delta": residual_delta.tolist(),
        "final_delta": final_delta.tolist(),
        "final_start_point": final_point.tolist(),
        "final_line_dir": None if final_line_dir is None else final_line_dir.tolist(),
        "final_line_point": final_point.tolist(),
        "final_plane_normals": None if final_plane_normals is None else final_plane_normals.tolist(),
        "infer_seconds": float(time.perf_counter() - start),
    }

    tool_point = None
    tool_line_dir = None
    if result_path is not None and result_path.exists():
        result = load_json(result_path)
        tool_point = np.asarray(result.get("weldStart", [0, 0, 0]), dtype=np.float32)
        line_coef12 = np.asarray(result.get("lineCoef12", [0, 0, 0, 0, 0, 0]), dtype=np.float32)
        if line_coef12.shape[0] >= 6:
            tool_line_dir = normalize_np(line_coef12[3:6])
            if not np.any(tool_line_dir):
                tool_line_dir = None
        row["result_ref_err"] = float(result.get("refErr", 0.0))
        row["target_valid"] = is_valid_point(tool_point)
        row["tool_line_dir"] = None if tool_line_dir is None else tool_line_dir.tolist()
        if row["target_valid"]:
            target_delta = tool_point - ref_point
            final_err = final_delta - target_delta
            row["target_start_point"] = tool_point.tolist()
            row["target_delta"] = target_delta.tolist()
            row["final_l2_mm"] = float(np.linalg.norm(final_err))
            row["final_mae_mm"] = float(np.mean(np.abs(final_err)))
            row.update({f"final_{k}": v for k, v in decompose_error(final_err, end_delta).items()})

    return row, raw_points, ref_point, end_delta, coarse_delta, final_delta, final_line_dir, tool_point, tool_line_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FineLocation two-stage raw point-cloud inference tool.")
    parser.add_argument("--cloud", type=Path, default=None, help="single cloud_*.pcd")
    parser.add_argument("--param", type=Path, default=None, help="single param*.txt")
    parser.add_argument("--result", type=Path, default=None, help="optional result_*.txt")
    parser.add_argument("--raw-dir", type=Path, default=None, help="batch directory with cloud/param/result files")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--output-jsonl", type=Path, default=None)
    parser.add_argument("--save-json", type=Path, default=None)
    parser.add_argument("--stage1-checkpoint", type=Path, default=resource_path("checkpoints/stage1_best_l2.pt"))
    parser.add_argument("--stage2-checkpoint", type=Path, default=resource_path("checkpoints/stage2_best_l2.pt"))
    parser.add_argument("--stage1-points", type=int, default=STAGE1_POINTS)
    parser.add_argument("--stage2-knn-points", type=int, default=STAGE2_KNN_POINTS)
    parser.add_argument(
        "--stage1-sampling",
        choices=["random", "fps"],
        default="random",
        help="Stage-1 downsampling. The bundled checkpoints were trained with random sampling.",
    )
    parser.add_argument("--sampling-seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--point-size", type=float, default=2.0)
    parser.add_argument("--marker-radius", type=float, default=4.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    stage1 = load_model(args.stage1_checkpoint.resolve(), device)
    stage2 = load_model(args.stage2_checkpoint.resolve(), device)

    if args.raw_dir is not None:
        samples = find_matching_files(args.raw_dir.resolve(), args.recursive)
        if not samples:
            raise SystemExit(f"no matching cloud/param/result files found under {args.raw_dir}")
        out = args.output_jsonl or (Path.cwd() / "fineloc_predictions.jsonl")
        out.parent.mkdir(parents=True, exist_ok=True)
        ok = 0
        with out.open("w", encoding="utf-8") as f:
            for i, (cloud, param, result) in enumerate(samples, 1):
                try:
                    row, *_ = infer_one(
                        cloud,
                        param,
                        result,
                        stage1,
                        stage2,
                        device,
                        args.stage1_points,
                        args.stage2_knn_points,
                        args.stage1_sampling,
                        args.sampling_seed,
                    )
                    row["ok"] = True
                    ok += 1
                    print(f"[{i}/{len(samples)}] ok {row['infer_seconds']:.3f}s {result.name}")
                except Exception as exc:
                    row = {"ok": False, "cloud": str(cloud), "param": str(param), "result": str(result), "error": str(exc)}
                    print(f"[{i}/{len(samples)}] fail {result.name}: {exc}")
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps({"output_jsonl": str(out.resolve()), "samples": len(samples), "ok": ok}, ensure_ascii=False, indent=2))
        return

    if args.cloud is None or args.param is None:
        raise SystemExit("single inference requires --cloud and --param, or use --raw-dir for batch inference")

    row, raw_points, ref_point, end_delta, coarse_delta, final_delta, final_line_dir, tool_point, tool_line_dir = infer_one(
        args.cloud.resolve(),
        args.param.resolve(),
        None if args.result is None else args.result.resolve(),
        stage1,
        stage2,
        device,
        args.stage1_points,
        args.stage2_knn_points,
        args.stage1_sampling,
        args.sampling_seed,
    )
    text = json.dumps(row, ensure_ascii=False, indent=2)
    print(text)
    if args.save_json is not None:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_json.write_text(text + "\n", encoding="utf-8")
    if args.visualize:
        visualize_prediction(raw_points, ref_point, end_delta, coarse_delta, final_delta, final_line_dir, tool_point, tool_line_dir, args.point_size, args.marker_radius)


if __name__ == "__main__":
    main()
