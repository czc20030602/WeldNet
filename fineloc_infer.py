#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

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

    def __init__(self, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = float(dropout)
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

    def forward(self, points: torch.Tensor, _params: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        xyz = points.transpose(1, 2)
        point_feat = self.point_encoder(xyz)
        global_feat = torch.max(point_feat, dim=2, keepdim=True)[0]
        global_feat = global_feat.expand(-1, -1, points.shape[1])
        point_context = torch.cat([point_feat, global_feat, xyz], dim=1)
        point_context = self.point_head(point_context)
        heatmap_logits = self.heatmap_head(point_context).squeeze(1)
        offsets = self.offset_head(point_context).transpose(1, 2)
        candidate_points = points + offsets
        weights = torch.softmax(heatmap_logits, dim=1)
        pred_delta = torch.sum(candidate_points * weights.unsqueeze(-1), dim=1)
        return {
            "pred_delta": pred_delta,
            "heatmap_logits": heatmap_logits,
            "offsets": offsets,
            "weights": weights,
        }


def load_model(checkpoint_path: Path, device: torch.device) -> PointHeatmapOffsetNet:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = PointHeatmapOffsetNet().to(device)
    state = checkpoint["model_state"]
    result = model.load_state_dict(state, strict=False)
    unexpected = [key for key in result.unexpected_keys if not key.startswith("line_head.")]
    if result.missing_keys or unexpected:
        raise RuntimeError(f"incompatible checkpoint: missing={result.missing_keys}, unexpected={unexpected}")
    model.eval()
    return model


@torch.no_grad()
def predict_delta_global(
    model: nn.Module,
    points_global_ref_frame: np.ndarray,
    end_delta: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    frame = line_frame_from_end_delta(end_delta.astype(np.float32))
    points_line = points_global_ref_frame.astype(np.float32) @ frame
    points = torch.from_numpy(points_line).unsqueeze(0).to(device)
    params = torch.zeros((1, 0), dtype=torch.float32, device=device)
    pred_local = model(points, params)["pred_delta"].squeeze(0).detach().cpu().numpy().astype(np.float32)
    return (pred_local @ frame.T).astype(np.float32)


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
    tool_point: np.ndarray | None,
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
    lines = [[0, 1], [0, 2], [0, 3]]
    colors = [[0.0, 0.75, 0.25], [1.0, 0.8, 0.05], [0.55, 0.2, 0.85]]
    geoms: list[o3d.geometry.Geometry] = [
        pcd,
        make_sphere(ref, marker_radius, [0.1, 0.3, 0.95]),
        make_sphere(pred, marker_radius, [0.0, 0.72, 0.22]),
        make_sphere(coarse, marker_radius * 0.8, [1.0, 0.8, 0.05]),
        make_sphere(end, marker_radius * 0.75, [0.55, 0.2, 0.85]),
    ]
    if tool_point is not None and is_valid_point(tool_point):
        tool = tool_point.astype(np.float32) - ref_point
        tool_idx = len(points)
        points.append(tool)
        lines.extend([[0, tool_idx], [tool_idx, 1]])
        colors.extend([[0.95, 0.15, 0.15], [0.0, 0.65, 1.0]])
        geoms.append(make_sphere(tool, marker_radius, [0.95, 0.12, 0.12]))
    geoms.insert(1, make_line(np.stack(points, axis=0), lines, colors))

    print("colors: gray cloud, blue reference, red tool output, green network output, yellow stage1, purple seam direction")
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
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    start = time.perf_counter()
    param = load_json(param_path)
    ref_point = np.asarray(param["startPos"], dtype=np.float32)
    end_delta = np.asarray(param["endPos1"], dtype=np.float32) - ref_point
    raw_points = read_pcd_xyz(cloud_path)

    stage1_cloud = fps_downsample(raw_points, stage1_points) - ref_point[None, :]
    coarse_delta = predict_delta_global(stage1_model, stage1_cloud, end_delta, device)
    refine_points, radius = crop_knn_around_delta(raw_points, ref_point, coarse_delta, stage2_knn_points)
    residual_delta = predict_delta_global(stage2_model, refine_points, end_delta, device)
    final_delta = coarse_delta + residual_delta
    final_point = ref_point + final_delta

    row: dict[str, Any] = {
        "cloud": str(cloud_path.resolve()),
        "param": str(param_path.resolve()),
        "result": "" if result_path is None else str(result_path.resolve()),
        "raw_point_count": int(raw_points.shape[0]),
        "stage1_points": int(stage1_points),
        "stage2_knn_points": int(stage2_knn_points),
        "stage2_knn_radius_mm": float(radius),
        "ref_point": ref_point.tolist(),
        "end_delta": end_delta.tolist(),
        "coarse_delta": coarse_delta.tolist(),
        "coarse_point": (ref_point + coarse_delta).tolist(),
        "residual_delta": residual_delta.tolist(),
        "final_delta": final_delta.tolist(),
        "final_start_point": final_point.tolist(),
        "infer_seconds": float(time.perf_counter() - start),
    }

    tool_point = None
    if result_path is not None and result_path.exists():
        result = load_json(result_path)
        tool_point = np.asarray(result.get("weldStart", [0, 0, 0]), dtype=np.float32)
        row["result_ref_err"] = float(result.get("refErr", 0.0))
        row["target_valid"] = is_valid_point(tool_point)
        if row["target_valid"]:
            target_delta = tool_point - ref_point
            final_err = final_delta - target_delta
            row["target_start_point"] = tool_point.tolist()
            row["target_delta"] = target_delta.tolist()
            row["final_l2_mm"] = float(np.linalg.norm(final_err))
            row["final_mae_mm"] = float(np.mean(np.abs(final_err)))
            row.update({f"final_{k}": v for k, v in decompose_error(final_err, end_delta).items()})

    return row, raw_points, ref_point, end_delta, coarse_delta, final_delta, tool_point


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
                    row, *_ = infer_one(cloud, param, result, stage1, stage2, device, args.stage1_points, args.stage2_knn_points)
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

    row, raw_points, ref_point, end_delta, coarse_delta, final_delta, tool_point = infer_one(
        args.cloud.resolve(),
        args.param.resolve(),
        None if args.result is None else args.result.resolve(),
        stage1,
        stage2,
        device,
        args.stage1_points,
        args.stage2_knn_points,
    )
    text = json.dumps(row, ensure_ascii=False, indent=2)
    print(text)
    if args.save_json is not None:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_json.write_text(text + "\n", encoding="utf-8")
    if args.visualize:
        visualize_prediction(raw_points, ref_point, end_delta, coarse_delta, final_delta, tool_point, args.point_size, args.marker_radius)


if __name__ == "__main__":
    main()
