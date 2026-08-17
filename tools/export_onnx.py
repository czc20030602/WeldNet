#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]


class PointHeatmapOffsetNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.point_encoder = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )
        self.point_head = nn.Sequential(
            nn.Conv1d(515, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Identity(),
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        self.heatmap_head = nn.Conv1d(128, 1, 1)
        self.offset_head = nn.Conv1d(128, 3, 1)
        self.line_head = nn.Sequential(nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Identity(), nn.Linear(128, 3))
        self.plane_head = nn.Sequential(nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Identity(), nn.Linear(128, 6))

    def forward(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        xyz = points.transpose(1, 2)
        point_feat = self.point_encoder(xyz)
        global_vec = torch.max(point_feat, dim=2)[0]
        global_feat = global_vec.unsqueeze(-1).expand(-1, -1, points.shape[1])
        point_context = self.point_head(torch.cat([point_feat, global_feat, xyz], dim=1))
        heatmap_logits = self.heatmap_head(point_context).squeeze(1)
        offsets = self.offset_head(point_context).transpose(1, 2)
        weights = torch.softmax(heatmap_logits, dim=1)
        pred_delta = torch.sum((points + offsets) * weights.unsqueeze(-1), dim=1)
        pred_line_dir = torch.nn.functional.normalize(self.line_head(global_vec), dim=1, eps=1e-6)
        pred_plane_normals = torch.nn.functional.normalize(
            self.plane_head(global_vec).view(-1, 2, 3), dim=2, eps=1e-6
        )
        return pred_delta, pred_line_dir, pred_plane_normals


def export_one(checkpoint: Path, output: Path, point_count: int, opset: int) -> None:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = PointHeatmapOffsetNet()
    result = model.load_state_dict(payload["model_state"], strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"checkpoint mismatch: missing={result.missing_keys}, unexpected={result.unexpected_keys}")
    model.eval()
    example = torch.zeros((1, point_count, 3), dtype=torch.float32)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (example,),
        output,
        input_names=["points"],
        output_names=["pred_delta", "pred_line_dir", "pred_plane_normals"],
        dynamic_axes={
            "points": {0: "batch", 1: "points"},
            "pred_delta": {0: "batch"},
            "pred_line_dir": {0: "batch"},
            "pred_plane_normals": {0: "batch"},
        },
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"exported {checkpoint.name} -> {output} ({output.stat().st_size / 1024 / 1024:.2f} MiB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export FineLocation checkpoints to ONNX.")
    parser.add_argument("--stage1-checkpoint", type=Path, default=ROOT / "checkpoints/stage1_best_l2.pt")
    parser.add_argument("--stage2-checkpoint", type=Path, default=ROOT / "checkpoints/stage2_best_l2.pt")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()
    export_one(args.stage1_checkpoint, args.output_dir / "stage1.onnx", 8192, args.opset)
    export_one(args.stage2_checkpoint, args.output_dir / "stage2.onnx", 16384, args.opset)


if __name__ == "__main__":
    main()
