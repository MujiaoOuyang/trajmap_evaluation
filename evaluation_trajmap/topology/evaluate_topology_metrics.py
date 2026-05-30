#!/usr/bin/env python3
"""Evaluate local topology metrics for vectorized lane-line shapefiles.

The evaluator follows a local lane-segment graph workflow:
  WGS84/projected shp -> metric CRS -> resampling -> local segmentation -> directed graph.

Each local lane segment is a graph node. Directed edges connect segments whose
end/start points are close and whose direction is consistent.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class RawLine:
    fid: str
    cls: str
    lane_type: str
    coords: np.ndarray
    length_m: float
    start: np.ndarray
    end: np.ndarray
    heading: float


@dataclass(frozen=True)
class LocalSegment:
    sid: int
    fid: str
    chain_id: int
    seg_idx: int
    cls: str
    lane_type: str
    points: np.ndarray
    length_m: float
    start: np.ndarray
    end: np.ndarray
    heading: float


def require_gis_imports() -> None:
    try:
        import geopandas as gpd  # noqa: F401
        import shapely  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Missing GIS dependency. Run with an environment that has geopandas/shapely, "
            "for example:\n"
            "  conda run -n gotrackit python scripts/evaluate_topology_metrics.py ...\n"
        ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local topology evaluation for vectorized HD-map lane-line shapefiles."
    )
    parser.add_argument("--gt", required=True, help="Ground-truth .shp path.")
    parser.add_argument("--pred", required=True, help="Predicted .shp path.")
    parser.add_argument("--class-field", default="type")
    parser.add_argument("--gt-class-field", default=None)
    parser.add_argument("--pred-class-field", default=None)
    parser.add_argument("--lane-type-field", default="lane_type")
    parser.add_argument("--gt-lane-type-field", default=None)
    parser.add_argument("--pred-lane-type-field", default=None)
    parser.add_argument("--lane-class", default="lane")
    parser.add_argument("--ignore-class", action="store_true")
    parser.add_argument("--gt-class-map", default=None)
    parser.add_argument("--pred-class-map", default=None)
    parser.add_argument("--sample-spacing", type=float, default=1.0)
    parser.add_argument(
        "--segmentation-mode",
        choices=["line", "trajectory-normal"],
        default="line",
        help="Local segmentation mode. 'trajectory-normal' uses shared reference-trajectory normal cuts for GT and prediction.",
    )
    parser.add_argument(
        "--trajectory-tum",
        default=None,
        help="Optional TUM trajectory txt used by --segmentation-mode trajectory-normal.",
    )
    parser.add_argument(
        "--trajectory-already-map-crs",
        action="store_true",
        help="Use TUM x/y directly in the map CRS instead of ENU conversion.",
    )
    parser.add_argument("--lat-ref", type=float, default=32.022252, help="ENU origin latitude.")
    parser.add_argument("--lon-ref", type=float, default=118.726685, help="ENU origin longitude.")
    parser.add_argument("--h-ref", type=float, default=8.135000, help="ENU origin height.")
    parser.add_argument(
        "--trajectory-normal-half-width",
        type=float,
        default=80.0,
        help="Half length of each trajectory-normal cutting line, in meters. Default: 80.",
    )
    parser.add_argument("--segment-lengths", nargs="+", type=float, default=[20.0, 30.0, 50.0])
    parser.add_argument("--primary-segment-length", type=float, default=30.0)
    parser.add_argument("--distance-threshold", type=float, default=2.0)
    parser.add_argument("--endpoint-threshold", type=float, default=2.0)
    parser.add_argument(
        "--cross-chain-endpoint-threshold",
        type=float,
        default=1.0,
        help="Endpoint threshold for edges between different chains. Same-chain adjacent segments are linked directly. Default: 1.0m.",
    )
    parser.add_argument(
        "--chain-merge-threshold",
        type=float,
        default=0.1,
        help="Endpoint threshold used to merge raw line fragments before local segmentation. Default: 0.1m.",
    )
    parser.add_argument("--angle-threshold", type=float, default=45.0)
    parser.add_argument("--length-error-threshold", type=float, default=0.3)
    parser.add_argument(
        "--apls-max-length",
        type=float,
        default=50.0,
        help="Backward-compatible single Local APLS range in meters. Default: 50.",
    )
    parser.add_argument(
        "--apls-max-lengths",
        nargs="+",
        type=float,
        default=[30.0, 50.0, 80.0],
        help="One or more Local APLS ranges in meters, e.g. 30 50 80.",
    )
    parser.add_argument("--break-spacing", type=float, default=2.0)
    parser.add_argument("--duplicate-threshold", type=float, default=2.0)
    parser.add_argument("--curvature-split-angle", type=float, default=45.0)
    parser.add_argument("--output-json", default="topology_eval_results.json")
    parser.add_argument("--output-csv", default="topology_eval_results.csv")
    parser.add_argument(
        "--output-plot",
        default="topology_eval_visualization.png",
        help="Optional PNG visualization of primary-segment GT/pred matching. Use empty string to disable.",
    )
    return parser.parse_args()


def normalize_class(value: Any) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, float) and math.isnan(value):
        return "unknown"
    return str(value).strip()


def parse_class_map(text: str | None) -> dict[str, str]:
    if not text:
        return {}
    mapping: dict[str, str] = {}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid class map item '{item}'. Expected old=new.")
        old, new = item.split("=", 1)
        mapping[normalize_class(old)] = normalize_class(new)
    return mapping


def choose_metric_crs(gt_gdf, pred_gdf):
    if gt_gdf.crs is None and pred_gdf.crs is None:
        raise ValueError("Both shapefiles have no CRS. WGS84/projected CRS is required.")
    base = gt_gdf if gt_gdf.crs is not None else pred_gdf
    if base.crs is not None and not base.crs.is_geographic:
        return base.crs
    try:
        return base.estimate_utm_crs()
    except Exception as exc:
        raise ValueError("Could not estimate a metric UTM CRS from WGS84 data.") from exc


def iter_line_parts(geom) -> Iterable[Any]:
    if geom is None or geom.is_empty:
        return
    if geom.geom_type == "LineString":
        yield geom
    elif geom.geom_type == "MultiLineString":
        yield from geom.geoms
    elif geom.geom_type == "GeometryCollection":
        for part in geom.geoms:
            yield from iter_line_parts(part)


def sample_line(line, spacing: float) -> np.ndarray:
    length = float(line.length)
    if length <= 0.0:
        coords = np.asarray(line.coords, dtype=float)
        return coords[:1, :2]
    n_steps = max(int(math.ceil(length / spacing)), 1)
    distances = np.linspace(0.0, length, n_steps + 1)
    points = [line.interpolate(float(d)) for d in distances]
    return np.asarray([(p.x, p.y) for p in points], dtype=float)


def resample_polyline(points: np.ndarray, spacing: float) -> np.ndarray:
    if len(points) <= 1:
        return points.copy()
    diffs = np.diff(points, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    total = float(seg_lengths.sum())
    if total <= 0:
        return points[:1].copy()
    distances = np.arange(0.0, total, spacing)
    if len(distances) == 0 or distances[-1] < total:
        distances = np.append(distances, total)
    cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    out = []
    seg_idx = 0
    for distance in distances:
        while seg_idx < len(seg_lengths) - 1 and cumulative[seg_idx + 1] < distance:
            seg_idx += 1
        denom = seg_lengths[seg_idx]
        if denom <= 0:
            out.append(points[seg_idx].copy())
        else:
            t = (distance - cumulative[seg_idx]) / denom
            out.append(points[seg_idx] + t * (points[seg_idx + 1] - points[seg_idx]))
    return np.asarray(out, dtype=float)


def read_tum_xy(path: Path) -> np.ndarray:
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 3:
        raise ValueError(f"{path} must have at least timestamp, x, y columns.")
    return data[:, 1:3].astype(float)


def read_tum_xyz(path: Path) -> np.ndarray:
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 4:
        raise ValueError(f"{path} must have at least timestamp, x, y, z columns.")
    return data[:, 1:4].astype(float)


def enu_to_map_crs(
    enu_xyz: np.ndarray,
    *,
    lat_ref: float,
    lon_ref: float,
    h_ref: float,
    target_crs,
) -> np.ndarray:
    from pyproj import Transformer

    lat0 = math.radians(lat_ref)
    lon0 = math.radians(lon_ref)
    transformer_to_ecef = Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True)
    transformer_from_ecef = Transformer.from_crs("EPSG:4978", target_crs, always_xy=True)
    x0, y0, z0 = transformer_to_ecef.transform(lon_ref, lat_ref, h_ref)

    east = enu_xyz[:, 0]
    north = enu_xyz[:, 1]
    up = enu_xyz[:, 2]
    sin_lat = math.sin(lat0)
    cos_lat = math.cos(lat0)
    sin_lon = math.sin(lon0)
    cos_lon = math.cos(lon0)

    dx = -sin_lon * east - sin_lat * cos_lon * north + cos_lat * cos_lon * up
    dy = cos_lon * east - sin_lat * sin_lon * north + cos_lat * sin_lon * up
    dz = cos_lat * north + sin_lat * up

    map_x, map_y, _ = transformer_from_ecef.transform(x0 + dx, y0 + dy, z0 + dz)
    return np.column_stack([map_x, map_y])


def load_reference_trajectory(args: argparse.Namespace, metric_crs) -> np.ndarray | None:
    if not args.trajectory_tum:
        return None
    path = Path(args.trajectory_tum)
    if args.trajectory_already_map_crs:
        return read_tum_xy(path)
    return enu_to_map_crs(
        read_tum_xyz(path),
        lat_ref=args.lat_ref,
        lon_ref=args.lon_ref,
        h_ref=args.h_ref,
        target_crs=metric_crs,
    )


def trajectory_normal_cutters(
    trajectory_xy: np.ndarray,
    station_spacing: float,
    half_width: float,
) -> list[Any]:
    from shapely.geometry import LineString

    stations = resample_polyline(trajectory_xy, station_spacing)
    if len(stations) < 2:
        return []
    cutters = []
    for idx, point in enumerate(stations):
        if idx == 0:
            tangent = stations[1] - stations[0]
        elif idx == len(stations) - 1:
            tangent = stations[-1] - stations[-2]
        else:
            tangent = stations[idx + 1] - stations[idx - 1]
        norm = float(np.linalg.norm(tangent))
        if norm <= 0:
            continue
        tangent = tangent / norm
        normal = np.asarray([-tangent[1], tangent[0]], dtype=float)
        p0 = point - normal * half_width
        p1 = point + normal * half_width
        cutters.append(LineString([(float(p0[0]), float(p0[1])), (float(p1[0]), float(p1[1]))]))
    return cutters


def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 == 0.0 or n2 == 0.0:
        return 180.0
    cosv = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return math.degrees(math.acos(cosv))


def heading_delta_deg(a: float, b: float) -> float:
    diff = abs((a - b + 180.0) % 360.0 - 180.0)
    return min(diff, 360.0 - diff)


def segment_heading(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    vec = points[-1] - points[0]
    if np.linalg.norm(vec) == 0:
        return 0.0
    return math.degrees(math.atan2(float(vec[1]), float(vec[0]))) % 360.0


def cut_distances_for_line(line, segment_length: float, curvature_split_angle: float) -> list[float]:
    length = float(line.length)
    cuts = {0.0, length}
    if length <= 0.0:
        return [0.0]
    n_segments = max(int(math.ceil(length / segment_length)), 1)
    for idx in range(1, n_segments):
        cuts.add(min(length, idx * segment_length))

    coords = list(line.coords)
    if len(coords) >= 3 and curvature_split_angle > 0:
        from shapely.geometry import Point

        for idx in range(1, len(coords) - 1):
            p0 = np.asarray(coords[idx - 1][:2], dtype=float)
            p1 = np.asarray(coords[idx][:2], dtype=float)
            p2 = np.asarray(coords[idx + 1][:2], dtype=float)
            turn = angle_between(p1 - p0, p2 - p1)
            if turn >= curvature_split_angle:
                cuts.add(float(line.project(Point(float(p1[0]), float(p1[1])))))
    return sorted(cut for cut in cuts if 0.0 <= cut <= length)


def add_intersection_cut_distances(line, cutter, cuts: set[float]) -> None:
    points = line.intersection(cutter)
    if points.is_empty:
        return
    geoms = getattr(points, "geoms", [points])
    from shapely.geometry import Point

    for geom in geoms:
        if geom.is_empty:
            continue
        if geom.geom_type == "Point":
            cuts.add(float(line.project(geom)))
        elif geom.geom_type in {"MultiPoint", "GeometryCollection"}:
            for part in geom.geoms:
                if part.geom_type == "Point":
                    cuts.add(float(line.project(part)))
        elif geom.geom_type in {"LineString", "LinearRing"}:
            coords = list(geom.coords)
            if coords:
                cuts.add(float(line.project(Point(coords[0]))))
                cuts.add(float(line.project(Point(coords[-1]))))


def cut_distances_for_line_with_trajectory_normals(
    line,
    trajectory_cutters: list[Any],
    curvature_split_angle: float,
) -> list[float]:
    length = float(line.length)
    cuts = {0.0, length}
    if length <= 0.0:
        return [0.0]
    for cutter in trajectory_cutters:
        add_intersection_cut_distances(line, cutter, cuts)

    coords = list(line.coords)
    if len(coords) >= 3 and curvature_split_angle > 0:
        from shapely.geometry import Point

        for idx in range(1, len(coords) - 1):
            p0 = np.asarray(coords[idx - 1][:2], dtype=float)
            p1 = np.asarray(coords[idx][:2], dtype=float)
            p2 = np.asarray(coords[idx + 1][:2], dtype=float)
            turn = angle_between(p1 - p0, p2 - p1)
            if turn >= curvature_split_angle:
                cuts.add(float(line.project(Point(float(p1[0]), float(p1[1])))))
    return sorted(cut for cut in cuts if 0.0 <= cut <= length)


def substring_line(line, start_d: float, end_d: float):
    from shapely.ops import substring

    if end_d <= start_d:
        return None
    part = substring(line, start_d, end_d)
    if part.is_empty or part.length <= 0.0:
        return None
    return part


def load_raw_lines(
    shp_path: Path,
    class_field: str,
    lane_type_field: str | None,
    metric_crs,
    *,
    lane_class: str,
    ignore_class: bool,
    class_map: dict[str, str],
) -> list[RawLine]:
    import geopandas as gpd

    gdf = gpd.read_file(shp_path)
    if gdf.crs is None:
        raise ValueError(f"{shp_path} has no CRS; cannot evaluate metric distances.")
    if not ignore_class and class_field not in gdf.columns:
        raise ValueError(f"Class field '{class_field}' not found in {shp_path}.")
    if lane_type_field and lane_type_field not in gdf.columns:
        lane_type_field = None
    gdf = gdf.to_crs(metric_crs)

    raw_lines: list[RawLine] = []
    for row_idx, row in gdf.iterrows():
        cls = "all" if ignore_class else normalize_class(row[class_field])
        cls = class_map.get(cls, cls)
        if not ignore_class and cls != lane_class:
            continue
        lane_type = normalize_class(row[lane_type_field]) if lane_type_field else "unknown"
        for part_idx, line in enumerate(iter_line_parts(row.geometry)):
            if line.length <= 0:
                continue
            coords = np.asarray(line.coords, dtype=float)[:, :2]
            if len(coords) < 2:
                continue
            raw_lines.append(
                RawLine(
                    fid=f"{row_idx}:{part_idx}",
                    cls=cls,
                    lane_type=lane_type,
                    coords=coords,
                    length_m=float(line.length),
                    start=coords[0],
                    end=coords[-1],
                    heading=segment_heading(coords),
                )
            )
    return raw_lines


def merge_raw_lines_to_chains(
    raw_lines: list[RawLine],
    merge_threshold: float,
    angle_threshold: float,
    *,
    respect_lane_type: bool = True,
) -> list[RawLine]:
    if not raw_lines:
        return []
    starts = np.asarray([line.start for line in raw_lines])
    tree = cKDTree(starts)
    candidates: list[tuple[float, int, int]] = []
    for i, line in enumerate(raw_lines):
        for j in tree.query_ball_point(line.end, r=merge_threshold):
            if i == j:
                continue
            other = raw_lines[j]
            if line.cls != other.cls:
                continue
            if respect_lane_type and line.lane_type != other.lane_type:
                continue
            angle = heading_delta_deg(line.heading, other.heading)
            if angle > angle_threshold:
                continue
            dist = float(np.linalg.norm(line.end - other.start))
            candidates.append((dist + 0.01 * angle, i, j))
    candidates.sort(key=lambda item: item[0])

    successor: dict[int, int] = {}
    predecessor: dict[int, int] = {}
    for _, i, j in candidates:
        if i in successor or j in predecessor:
            continue
        successor[i] = j
        predecessor[j] = i

    chains: list[RawLine] = []
    visited: set[int] = set()
    starts_idx = [idx for idx in range(len(raw_lines)) if idx not in predecessor]
    starts_idx.extend(idx for idx in range(len(raw_lines)) if idx in predecessor)

    for start_idx in starts_idx:
        if start_idx in visited:
            continue
        ids = []
        current = start_idx
        while current not in visited:
            visited.add(current)
            ids.append(current)
            if current not in successor:
                break
            current = successor[current]
        if not ids:
            continue

        coords_parts = []
        for part_num, raw_idx in enumerate(ids):
            coords = raw_lines[raw_idx].coords
            coords_parts.append(coords if part_num == 0 else coords[1:])
        coords = np.vstack(coords_parts)
        if len(coords) < 2:
            continue
        length = float(np.linalg.norm(np.diff(coords, axis=0), axis=1).sum())
        first = raw_lines[ids[0]]
        chains.append(
            RawLine(
                fid="+".join(raw_lines[idx].fid for idx in ids),
                cls=first.cls,
                lane_type=first.lane_type,
                coords=coords,
                length_m=length,
                start=coords[0],
                end=coords[-1],
                heading=segment_heading(coords),
            )
        )
    return chains


def load_local_segments(
    shp_path: Path,
    class_field: str,
    lane_type_field: str | None,
    metric_crs,
    *,
    lane_class: str,
    ignore_class: bool,
    class_map: dict[str, str],
    sample_spacing: float,
    segment_length: float,
    curvature_split_angle: float,
    chain_merge_threshold: float,
    angle_threshold: float,
    segmentation_mode: str,
    trajectory_cutters: list[Any] | None,
) -> list[LocalSegment]:
    from shapely.geometry import LineString

    raw_lines = load_raw_lines(
        shp_path,
        class_field,
        lane_type_field,
        lane_class=lane_class,
        ignore_class=ignore_class,
        class_map=class_map,
        metric_crs=metric_crs,
    )
    chains = merge_raw_lines_to_chains(
        raw_lines,
        chain_merge_threshold,
        angle_threshold,
        respect_lane_type=not ignore_class,
    )

    segments: list[LocalSegment] = []
    sid = 0
    for chain_idx, chain in enumerate(chains):
        line = LineString(chain.coords)
        if segmentation_mode == "trajectory-normal":
            cuts = cut_distances_for_line_with_trajectory_normals(
                line,
                trajectory_cutters or [],
                curvature_split_angle,
            )
            if len(cuts) <= 2 and float(line.length) > 1.5 * segment_length:
                cuts = cut_distances_for_line(line, segment_length, curvature_split_angle)
        else:
            cuts = cut_distances_for_line(line, segment_length, curvature_split_angle)
        for seg_idx, (start_d, end_d) in enumerate(zip(cuts[:-1], cuts[1:])):
            part = substring_line(line, start_d, end_d)
            if part is None:
                continue
            points = sample_line(part, sample_spacing)
            if len(points) < 2:
                continue
            segments.append(
                LocalSegment(
                    sid=sid,
                    fid=f"{chain_idx}:{seg_idx}",
                    chain_id=chain_idx,
                    seg_idx=seg_idx,
                    cls=chain.cls,
                    lane_type=chain.lane_type,
                    points=points,
                    length_m=float(part.length),
                    start=points[0],
                    end=points[-1],
                    heading=segment_heading(points),
                )
            )
            sid += 1
    return segments


def chamfer_mean(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return float("inf")
    tree_b = cKDTree(b)
    tree_a = cKDTree(a)
    d_ab = float(tree_b.query(a, k=1, workers=-1)[0].mean())
    d_ba = float(tree_a.query(b, k=1, workers=-1)[0].mean())
    return 0.5 * (d_ab + d_ba)


def build_edges(
    segments: list[LocalSegment],
    endpoint_threshold: float,
    angle_threshold: float,
    cross_chain_endpoint_threshold: float,
) -> set[tuple[int, int]]:
    if not segments:
        return set()
    by_chain_seg = {(seg.chain_id, seg.seg_idx): seg.sid for seg in segments}
    edges: set[tuple[int, int]] = set()

    for seg in segments:
        next_sid = by_chain_seg.get((seg.chain_id, seg.seg_idx + 1))
        if next_sid is not None:
            edges.add((seg.sid, next_sid))

    starts = np.asarray([seg.start for seg in segments])
    tree = cKDTree(starts)
    for seg in segments:
        for other_idx in tree.query_ball_point(seg.end, r=cross_chain_endpoint_threshold):
            other = segments[other_idx]
            if seg.sid == other.sid:
                continue
            if seg.chain_id == other.chain_id:
                continue
            if float(np.linalg.norm(seg.end - other.start)) > endpoint_threshold:
                continue
            if heading_delta_deg(seg.heading, other.heading) <= angle_threshold:
                edges.add((seg.sid, other.sid))
    return edges


def match_segments(
    pred: list[LocalSegment],
    gt: list[LocalSegment],
    distance_threshold: float,
    endpoint_threshold: float,
    angle_threshold: float,
) -> tuple[dict[int, int], dict[int, int], list[dict[str, Any]]]:
    if not pred or not gt:
        return {}, {}, []
    gt_centers = np.asarray([(seg.start + seg.end) * 0.5 for seg in gt])
    pred_centers = np.asarray([(seg.start + seg.end) * 0.5 for seg in pred])
    gt_radius = np.asarray([np.linalg.norm(seg.points - gt_centers[i], axis=1).max() for i, seg in enumerate(gt)])
    pred_radius = np.asarray(
        [np.linalg.norm(seg.points - pred_centers[i], axis=1).max() for i, seg in enumerate(pred)]
    )
    gt_tree = cKDTree(gt_centers)
    max_gt_radius = float(gt_radius.max(initial=0.0))
    candidates: list[tuple[float, int, int, float, float, float]] = []
    for pi, pred_seg in enumerate(pred):
        search_radius = distance_threshold + endpoint_threshold + pred_radius[pi] + max_gt_radius
        for gi in gt_tree.query_ball_point(pred_centers[pi], r=search_radius):
            gt_seg = gt[gi]
            start_d = float(np.linalg.norm(pred_seg.start - gt_seg.start))
            end_d = float(np.linalg.norm(pred_seg.end - gt_seg.end))
            if start_d >= endpoint_threshold or end_d >= endpoint_threshold:
                continue
            angle_d = heading_delta_deg(pred_seg.heading, gt_seg.heading)
            if angle_d >= angle_threshold:
                continue
            cd = chamfer_mean(pred_seg.points, gt_seg.points)
            if cd >= distance_threshold:
                continue
            score = cd + 0.25 * (start_d + end_d) + 0.01 * angle_d
            candidates.append((score, pi, gi, cd, start_d, end_d))
    candidates.sort(key=lambda item: item[0])

    pred_to_gt: dict[int, int] = {}
    gt_to_pred: dict[int, int] = {}
    details: list[dict[str, Any]] = []
    for score, pi, gi, cd, start_d, end_d in candidates:
        pred_sid = pred[pi].sid
        gt_sid = gt[gi].sid
        if pred_sid in pred_to_gt or gt_sid in gt_to_pred:
            continue
        pred_to_gt[pred_sid] = gt_sid
        gt_to_pred[gt_sid] = pred_sid
        details.append(
            {
                "pred": pred_sid,
                "gt": gt_sid,
                "score": float(score),
                "chamfer_mean_m": float(cd),
                "start_error_m": float(start_d),
                "end_error_m": float(end_d),
            }
        )
    return pred_to_gt, gt_to_pred, details


def assign_pred_to_gt_many_to_one_strict(
    pred: list[LocalSegment],
    gt: list[LocalSegment],
    distance_threshold: float,
    endpoint_threshold: float,
    angle_threshold: float,
) -> dict[int, int]:
    if not pred or not gt:
        return {}
    gt_centers = np.asarray([(seg.start + seg.end) * 0.5 for seg in gt])
    pred_centers = np.asarray([(seg.start + seg.end) * 0.5 for seg in pred])
    gt_radius = np.asarray([np.linalg.norm(seg.points - gt_centers[i], axis=1).max() for i, seg in enumerate(gt)])
    pred_radius = np.asarray(
        [np.linalg.norm(seg.points - pred_centers[i], axis=1).max() for i, seg in enumerate(pred)]
    )
    gt_tree = cKDTree(gt_centers)
    max_gt_radius = float(gt_radius.max(initial=0.0))
    pred_to_gt: dict[int, int] = {}

    for pi, pred_seg in enumerate(pred):
        search_radius = distance_threshold + endpoint_threshold + pred_radius[pi] + max_gt_radius
        best: tuple[float, int] | None = None
        for gi in gt_tree.query_ball_point(pred_centers[pi], r=search_radius):
            gt_seg = gt[gi]
            start_d = float(np.linalg.norm(pred_seg.start - gt_seg.start))
            end_d = float(np.linalg.norm(pred_seg.end - gt_seg.end))
            if start_d >= endpoint_threshold or end_d >= endpoint_threshold:
                continue
            angle_d = heading_delta_deg(pred_seg.heading, gt_seg.heading)
            if angle_d >= angle_threshold:
                continue
            cd = chamfer_mean(pred_seg.points, gt_seg.points)
            if cd >= distance_threshold:
                continue
            score = cd + 0.25 * (start_d + end_d) + 0.01 * angle_d
            if best is None or score < best[0]:
                best = (score, gt_seg.sid)
        if best is not None:
            pred_to_gt[pred_seg.sid] = best[1]
    return pred_to_gt


def adjacency(edges: set[tuple[int, int]]) -> dict[int, list[int]]:
    adj: dict[int, list[int]] = defaultdict(list)
    for src, dst in edges:
        adj[src].append(dst)
    return adj


def topology_f1(
    gt_edges: set[tuple[int, int]],
    pred_edges: set[tuple[int, int]],
    pred_to_gt_for_precision: dict[int, int],
    gt_to_pred: dict[int, int],
) -> dict[str, Any]:
    tp_pred = 0
    for src, dst in pred_edges:
        if (
            src in pred_to_gt_for_precision
            and dst in pred_to_gt_for_precision
            and (pred_to_gt_for_precision[src], pred_to_gt_for_precision[dst]) in gt_edges
        ):
            tp_pred += 1
    tp_gt = 0
    for src, dst in gt_edges:
        if src in gt_to_pred and dst in gt_to_pred and (gt_to_pred[src], gt_to_pred[dst]) in pred_edges:
            tp_gt += 1
    precision = tp_pred / len(pred_edges) if pred_edges else 0.0
    recall = tp_gt / len(gt_edges) if gt_edges else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "tp_pred_edges": tp_pred,
        "tp_gt_edges": tp_gt,
        "num_pred_edges": len(pred_edges),
        "num_gt_edges": len(gt_edges),
        "pred_segment_assignment_mode": "many_to_one",
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def concat_points(ids: list[int], by_sid: dict[int, LocalSegment]) -> np.ndarray:
    parts = []
    for idx, sid in enumerate(ids):
        pts = by_sid[sid].points
        parts.append(pts if idx == 0 else pts[1:])
    return np.vstack(parts) if parts else np.empty((0, 2))


def shortest_path(
    adj: dict[int, list[int]],
    by_sid: dict[int, LocalSegment],
    src: int,
    dst: int,
    max_length: float,
) -> tuple[list[int] | None, float | None]:
    queue = deque([(src, [src], by_sid[src].length_m)])
    best_length = {src: by_sid[src].length_m}
    while queue:
        node, path, length = queue.popleft()
        if node == dst:
            return path, length
        for nxt in adj.get(node, []):
            next_length = length + by_sid[nxt].length_m
            if next_length > max_length:
                continue
            if next_length >= best_length.get(nxt, float("inf")):
                continue
            best_length[nxt] = next_length
            queue.append((nxt, path + [nxt], next_length))
    return None, None


def shortest_length(
    adj: dict[int, list[int]],
    by_sid: dict[int, LocalSegment],
    src: int,
    dst: int,
    max_length: float,
) -> float | None:
    queue = deque([(src, by_sid[src].length_m)])
    best_length = {src: by_sid[src].length_m}
    while queue:
        node, length = queue.popleft()
        if node == dst:
            return length
        for nxt in adj.get(node, []):
            next_length = length + by_sid[nxt].length_m
            if next_length > max_length:
                continue
            if next_length >= best_length.get(nxt, float("inf")):
                continue
            best_length[nxt] = next_length
            queue.append((nxt, next_length))
    return None


def path_based_recall(
    gt_segments: list[LocalSegment],
    pred_segments: list[LocalSegment],
    gt_edges: set[tuple[int, int]],
    pred_edges: set[tuple[int, int]],
    gt_to_pred: dict[int, int],
    distance_threshold: float,
    length_error_threshold: float,
    max_length: float,
) -> dict[str, Any]:
    gt_by_sid = {seg.sid: seg for seg in gt_segments}
    pred_by_sid = {seg.sid: seg for seg in pred_segments}
    pred_adj = adjacency(pred_edges)
    tp = 0
    tested = 0
    for src, dst in gt_edges:
        if src not in gt_to_pred or dst not in gt_to_pred:
            continue
        tested += 1
        pred_src = gt_to_pred[src]
        pred_dst = gt_to_pred[dst]
        gt_path_points = concat_points([src, dst], gt_by_sid)
        gt_len = gt_by_sid[src].length_m + gt_by_sid[dst].length_m
        search_limit = max(max_length, gt_len * (1.0 + length_error_threshold))
        path, pred_len = shortest_path(pred_adj, pred_by_sid, pred_src, pred_dst, search_limit)
        if path is None or pred_len is None or gt_len <= 0:
            continue
        pred_path_points = concat_points(path, pred_by_sid)
        length_error = abs(pred_len - gt_len) / gt_len
        if length_error <= length_error_threshold and chamfer_mean(pred_path_points, gt_path_points) < distance_threshold:
            tp += 1
    return {
        "tp_path_edges": tp,
        "num_gt_edges": len(gt_edges),
        "matched_endpoint_gt_edges": tested,
        "recall": tp / len(gt_edges) if gt_edges else 0.0,
    }


def all_pairs_lengths_within(
    segments: list[LocalSegment], edges: set[tuple[int, int]], max_length: float
) -> dict[tuple[int, int], float]:
    by_sid = {seg.sid: seg for seg in segments}
    adj = adjacency(edges)
    lengths: dict[tuple[int, int], float] = {}
    for src in by_sid:
        queue = deque([(src, by_sid[src].length_m)])
        best = {src: by_sid[src].length_m}
        while queue:
            node, length = queue.popleft()
            if node != src and length <= max_length:
                lengths[(src, node)] = length
            for nxt in adj.get(node, []):
                next_length = length + by_sid[nxt].length_m
                if next_length > max_length:
                    continue
                if next_length >= best.get(nxt, float("inf")):
                    continue
                best[nxt] = next_length
                queue.append((nxt, next_length))
    return lengths


def local_apls(
    gt_segments: list[LocalSegment],
    pred_segments: list[LocalSegment],
    gt_edges: set[tuple[int, int]],
    pred_edges: set[tuple[int, int]],
    gt_to_pred: dict[int, int],
    max_length: float,
) -> dict[str, Any]:
    gt_lengths = all_pairs_lengths_within(gt_segments, gt_edges, max_length)
    pred_by_sid = {seg.sid: seg for seg in pred_segments}
    pred_adj = adjacency(pred_edges)
    scores = []
    unreachable = 0
    unmatched = 0
    for (gt_src, gt_dst), gt_len in gt_lengths.items():
        if gt_src not in gt_to_pred or gt_dst not in gt_to_pred:
            unmatched += 1
            scores.append(0.0)
            continue
        pred_len = shortest_length(
            pred_adj,
            pred_by_sid,
            gt_to_pred[gt_src],
            gt_to_pred[gt_dst],
            max_length * 2.0,
        )
        if pred_len is None:
            unreachable += 1
            scores.append(0.0)
            continue
        scores.append(1.0 - min(1.0, abs(pred_len - gt_len) / gt_len))
    return {
        "max_length_m": max_length,
        "num_gt_pairs": len(scores),
        "unmatched_pairs": unmatched,
        "unreachable_pairs": unreachable,
        "score": float(np.mean(scores)) if scores else 0.0,
    }


def apls_key(max_length: float) -> str:
    if float(max_length).is_integer():
        return f"{int(max_length)}m"
    return f"{max_length:g}m"


def break_rate(
    gt_segments: list[LocalSegment],
    pred_segments: list[LocalSegment],
    break_spacing: float,
    distance_threshold: float,
    angle_threshold: float,
) -> dict[str, Any]:
    if not gt_segments:
        return {
            "definition": "fragmentation_rate",
            "segmental_coverage": 0.0,
            "mean_gt_segment_length_m": 0.0,
            "mean_pred_segment_length_m": 0.0,
            "break_rate": 1.0,
        }
    if not pred_segments:
        return {
            "definition": "fragmentation_rate",
            "segmental_coverage": 0.0,
            "mean_gt_segment_length_m": float(np.mean([seg.length_m for seg in gt_segments])),
            "mean_pred_segment_length_m": 0.0,
            "break_rate": 1.0,
        }
    pred_points = np.vstack([seg.points for seg in pred_segments])
    pred_headings = np.concatenate([np.full(len(seg.points), seg.heading) for seg in pred_segments])
    tree = cKDTree(pred_points)

    covered_len = 0.0
    total_len = 0.0
    for gt_seg in gt_segments:
        n_steps = max(int(math.ceil(gt_seg.length_m / break_spacing)), 1)
        distances = np.linspace(0.0, gt_seg.length_m, n_steps + 1)
        mids = 0.5 * (distances[:-1] + distances[1:])
        chunk_lengths = np.diff(distances)
        for chunk_idx, mid in enumerate(mids):
            idx = min(int(round(mid / gt_seg.length_m * (len(gt_seg.points) - 1))), len(gt_seg.points) - 1)
            point = gt_seg.points[idx]
            nearby = tree.query_ball_point(point, r=distance_threshold)
            ok = any(heading_delta_deg(gt_seg.heading, float(pred_headings[pi])) <= angle_threshold for pi in nearby)
            if ok:
                covered_len += float(chunk_lengths[chunk_idx])
        total_len += float(chunk_lengths.sum())

    coverage = covered_len / total_len if total_len else 0.0
    mean_gt_length = float(np.mean([seg.length_m for seg in gt_segments]))
    mean_pred_length = float(np.mean([seg.length_m for seg in pred_segments]))
    fragmentation = 1.0 - min(1.0, mean_pred_length / mean_gt_length) if mean_gt_length > 0 else 1.0
    return {
        "definition": "fragmentation_rate",
        "segmental_coverage": coverage,
        "mean_gt_segment_length_m": mean_gt_length,
        "mean_pred_segment_length_m": mean_pred_length,
        "break_rate": fragmentation,
    }


def duplicate_rate(
    pred_segments: list[LocalSegment],
    gt_segments: list[LocalSegment],
    distance_threshold: float,
    angle_threshold: float,
) -> dict[str, Any]:
    total_length = float(sum(seg.length_m for seg in pred_segments))
    if total_length <= 0.0:
        return {
            "definition": "global_pred_duplicate_rate",
            "duplicate_length_m": 0.0,
            "pred_total_length_m": 0.0,
            "duplicate_rate": 0.0,
        }
    dup_ids: set[int] = set()

    centers = np.asarray([(seg.start + seg.end) * 0.5 for seg in pred_segments])
    radii = np.asarray(
        [np.linalg.norm(seg.points - centers[i], axis=1).max() for i, seg in enumerate(pred_segments)]
    )
    tree = cKDTree(centers)
    max_radius = float(radii.max(initial=0.0))
    by_sid = {seg.sid: idx for idx, seg in enumerate(pred_segments)}
    sorted_segments = sorted(pred_segments, key=lambda seg: seg.length_m, reverse=True)
    for base in sorted_segments:
        base_idx = by_sid[base.sid]
        if base.sid in dup_ids:
            continue
        search_radius = float(radii[base_idx] + max_radius + distance_threshold)
        for other_idx in tree.query_ball_point(centers[base_idx], r=search_radius):
            other = pred_segments[other_idx]
            if other.sid == base.sid or other.sid in dup_ids:
                continue
            if other.length_m > base.length_m:
                continue
            if heading_delta_deg(base.heading, other.heading) > angle_threshold:
                continue
            if chamfer_mean(base.points, other.points) < distance_threshold:
                dup_ids.add(other.sid)
    dup_length = float(sum(seg.length_m for seg in pred_segments if seg.sid in dup_ids))
    return {
        "definition": "global_pred_duplicate_rate",
        "duplicate_segments": len(dup_ids),
        "duplicate_length_m": dup_length,
        "pred_total_length_m": total_length,
        "duplicate_rate": dup_length / total_length,
    }


def assign_pred_to_nearest_gt_many_to_one(
    pred_segments: list[LocalSegment],
    gt_segments: list[LocalSegment],
    distance_threshold: float,
    angle_threshold: float,
) -> dict[int, list[LocalSegment]]:
    assignments: dict[int, list[LocalSegment]] = defaultdict(list)
    if not pred_segments or not gt_segments:
        return assignments

    gt_centers = np.asarray([(seg.start + seg.end) * 0.5 for seg in gt_segments])
    gt_radius = np.asarray(
        [np.linalg.norm(seg.points - gt_centers[i], axis=1).max() for i, seg in enumerate(gt_segments)]
    )
    tree = cKDTree(gt_centers)

    for pred_seg in pred_segments:
        pred_center = (pred_seg.start + pred_seg.end) * 0.5
        pred_radius = float(np.linalg.norm(pred_seg.points - pred_center, axis=1).max())
        search_radius = distance_threshold + pred_radius + float(gt_radius.max(initial=0.0))
        best_sid = None
        best_cd = float("inf")
        for gi in tree.query_ball_point(pred_center, r=search_radius):
            gt_seg = gt_segments[gi]
            angle_d = heading_delta_deg(pred_seg.heading, gt_seg.heading)
            if angle_d > angle_threshold:
                continue
            cd = chamfer_mean(pred_seg.points, gt_seg.points)
            if cd < best_cd:
                best_cd = cd
                best_sid = gt_seg.sid
        if best_sid is not None and best_cd < distance_threshold:
            assignments[best_sid].append(pred_seg)
    return assignments


def evaluate_for_segment_length(
    args: argparse.Namespace,
    metric_crs,
    gt_path: Path,
    pred_path: Path,
    segment_length: float,
    apls_max_lengths: list[float],
    trajectory_xy: np.ndarray | None,
) -> dict[str, Any]:
    gt_field = args.gt_class_field or args.class_field
    pred_field = args.pred_class_field or args.class_field
    gt_lane_type_field = args.gt_lane_type_field or args.lane_type_field
    pred_lane_type_field = args.pred_lane_type_field or args.lane_type_field
    lane_class = normalize_class(args.lane_class)
    chain_merge_threshold = args.chain_merge_threshold
    if args.segmentation_mode == "trajectory-normal":
        if trajectory_xy is None:
            raise ValueError("--trajectory-tum is required for --segmentation-mode trajectory-normal.")
        trajectory_cutters = trajectory_normal_cutters(
            trajectory_xy,
            segment_length,
            args.trajectory_normal_half_width,
        )
    else:
        trajectory_cutters = None
    gt_segments = load_local_segments(
        gt_path,
        gt_field,
        gt_lane_type_field,
        metric_crs,
        lane_class=lane_class,
        ignore_class=args.ignore_class,
        class_map=parse_class_map(args.gt_class_map),
        sample_spacing=args.sample_spacing,
        segment_length=segment_length,
        curvature_split_angle=args.curvature_split_angle,
        chain_merge_threshold=chain_merge_threshold,
        angle_threshold=args.angle_threshold,
        segmentation_mode=args.segmentation_mode,
        trajectory_cutters=trajectory_cutters,
    )
    pred_segments = load_local_segments(
        pred_path,
        pred_field,
        pred_lane_type_field,
        metric_crs,
        lane_class=lane_class,
        ignore_class=args.ignore_class,
        class_map=parse_class_map(args.pred_class_map),
        sample_spacing=args.sample_spacing,
        segment_length=segment_length,
        curvature_split_angle=args.curvature_split_angle,
        chain_merge_threshold=chain_merge_threshold,
        angle_threshold=args.angle_threshold,
        segmentation_mode=args.segmentation_mode,
        trajectory_cutters=trajectory_cutters,
    )
    gt_edges = build_edges(
        gt_segments,
        args.endpoint_threshold,
        args.angle_threshold,
        args.cross_chain_endpoint_threshold,
    )
    pred_edges = build_edges(
        pred_segments,
        args.endpoint_threshold,
        args.angle_threshold,
        args.cross_chain_endpoint_threshold,
    )
    pred_to_gt, gt_to_pred, match_details = match_segments(
        pred_segments,
        gt_segments,
        args.distance_threshold,
        args.endpoint_threshold,
        args.angle_threshold,
    )
    pred_to_gt_many_to_one = assign_pred_to_gt_many_to_one_strict(
        pred_segments,
        gt_segments,
        args.distance_threshold,
        args.endpoint_threshold,
        args.angle_threshold,
    )

    topo = topology_f1(gt_edges, pred_edges, pred_to_gt_many_to_one, gt_to_pred)
    path_recall = path_based_recall(
        gt_segments,
        pred_segments,
        gt_edges,
        pred_edges,
        gt_to_pred,
        args.distance_threshold,
        args.length_error_threshold,
        max(apls_max_lengths),
    )
    direct_edge_recall = topo["recall"]
    topo["direct_edge_recall"] = direct_edge_recall
    topo["tp_direct_gt_edges"] = topo["tp_gt_edges"]
    topo["tp_gt_edges"] = path_recall["tp_path_edges"]
    topo["recall"] = path_recall["recall"]
    topo["recall_definition"] = "path_based_local_recall"
    topo["f1"] = (
        2 * topo["precision"] * topo["recall"] / (topo["precision"] + topo["recall"])
        if (topo["precision"] + topo["recall"])
        else 0.0
    )
    apls_by_range = {
        apls_key(max_length): local_apls(
            gt_segments,
            pred_segments,
            gt_edges,
            pred_edges,
            gt_to_pred,
            max_length,
        )
        for max_length in apls_max_lengths
    }
    br = break_rate(
        gt_segments,
        pred_segments,
        args.break_spacing,
        args.distance_threshold,
        args.angle_threshold,
    )
    dr = duplicate_rate(pred_segments, gt_segments, args.duplicate_threshold, args.angle_threshold)

    return {
        "segment_length_m": segment_length,
        "num_gt_segments": len(gt_segments),
        "num_pred_segments": len(pred_segments),
        "gt_total_length_m": float(sum(seg.length_m for seg in gt_segments)),
        "pred_total_length_m": float(sum(seg.length_m for seg in pred_segments)),
        "num_gt_edges": len(gt_edges),
        "num_pred_edges": len(pred_edges),
        "num_matched_segments": len(pred_to_gt),
        "num_topology_pred_assigned_segments": len(pred_to_gt_many_to_one),
        "segment_match_recall": len(gt_to_pred) / len(gt_segments) if gt_segments else 0.0,
        "segment_match_precision": len(pred_to_gt) / len(pred_segments) if pred_segments else 0.0,
        "local_topology_f1": topo,
        "path_based_local_recall": path_recall,
        "local_directed_apls": apls_by_range,
        "break_rate": br,
        "duplicate_rate": dr,
        "match_summary": {
            "mean_chamfer_m": float(np.mean([m["chamfer_mean_m"] for m in match_details])) if match_details else None,
            "median_chamfer_m": float(np.median([m["chamfer_mean_m"] for m in match_details])) if match_details else None,
        },
    }


def write_csv(path: Path, report: dict[str, Any]) -> None:
    rows = []
    apls_keys = [apls_key(v) for v in report["inputs"]["apls_max_lengths_m"]]
    for item in report["segment_length_results"]:
        row = {
            "segment_length_m": item["segment_length_m"],
            "num_gt_segments": item["num_gt_segments"],
            "num_pred_segments": item["num_pred_segments"],
            "num_gt_edges": item["num_gt_edges"],
            "num_pred_edges": item["num_pred_edges"],
            "num_matched_segments": item["num_matched_segments"],
            "segment_match_precision": item["segment_match_precision"],
            "segment_match_recall": item["segment_match_recall"],
            "topology_precision": item["local_topology_f1"]["precision"],
            "topology_recall": item["local_topology_f1"]["recall"],
            "topology_f1": item["local_topology_f1"]["f1"],
            "direct_edge_recall": item["local_topology_f1"]["direct_edge_recall"],
            "segmental_coverage": item["break_rate"]["segmental_coverage"],
            "break_rate": item["break_rate"]["break_rate"],
            "duplicate_rate": item["duplicate_rate"]["duplicate_rate"],
            "matched_mean_chamfer_m": item["match_summary"]["mean_chamfer_m"],
            "matched_median_chamfer_m": item["match_summary"]["median_chamfer_m"],
        }
        for key in apls_keys:
            row[f"local_apls_{key}"] = item["local_directed_apls"][key]["score"]
        rows.append(row)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def segments_to_lines(segments: list[LocalSegment]) -> list[np.ndarray]:
    return [seg.points for seg in segments if len(seg.points) >= 2]


def load_segments_for_plot(
    args: argparse.Namespace,
    metric_crs,
    gt_path: Path,
    pred_path: Path,
    segment_length: float,
    trajectory_xy: np.ndarray | None,
) -> tuple[list[LocalSegment], list[LocalSegment], dict[int, int], dict[int, int]]:
    gt_field = args.gt_class_field or args.class_field
    pred_field = args.pred_class_field or args.class_field
    gt_lane_type_field = args.gt_lane_type_field or args.lane_type_field
    pred_lane_type_field = args.pred_lane_type_field or args.lane_type_field
    lane_class = normalize_class(args.lane_class)
    if args.segmentation_mode == "trajectory-normal":
        if trajectory_xy is None:
            raise ValueError("--trajectory-tum is required for --segmentation-mode trajectory-normal.")
        trajectory_cutters = trajectory_normal_cutters(
            trajectory_xy,
            segment_length,
            args.trajectory_normal_half_width,
        )
    else:
        trajectory_cutters = None

    gt_segments = load_local_segments(
        gt_path,
        gt_field,
        gt_lane_type_field,
        metric_crs,
        lane_class=lane_class,
        ignore_class=args.ignore_class,
        class_map=parse_class_map(args.gt_class_map),
        sample_spacing=args.sample_spacing,
        segment_length=segment_length,
        curvature_split_angle=args.curvature_split_angle,
        chain_merge_threshold=args.chain_merge_threshold,
        angle_threshold=args.angle_threshold,
        segmentation_mode=args.segmentation_mode,
        trajectory_cutters=trajectory_cutters,
    )
    pred_segments = load_local_segments(
        pred_path,
        pred_field,
        pred_lane_type_field,
        metric_crs,
        lane_class=lane_class,
        ignore_class=args.ignore_class,
        class_map=parse_class_map(args.pred_class_map),
        sample_spacing=args.sample_spacing,
        segment_length=segment_length,
        curvature_split_angle=args.curvature_split_angle,
        chain_merge_threshold=args.chain_merge_threshold,
        angle_threshold=args.angle_threshold,
        segmentation_mode=args.segmentation_mode,
        trajectory_cutters=trajectory_cutters,
    )
    pred_to_gt, gt_to_pred, _ = match_segments(
        pred_segments,
        gt_segments,
        args.distance_threshold,
        args.endpoint_threshold,
        args.angle_threshold,
    )
    return gt_segments, pred_segments, pred_to_gt, gt_to_pred


def write_match_plot(
    path: Path,
    *,
    gt_segments: list[LocalSegment],
    pred_segments: list[LocalSegment],
    pred_to_gt: dict[int, int],
    gt_to_pred: dict[int, int],
    trajectory_xy: np.ndarray | None,
    title: str,
) -> Path:
    try:
        import matplotlib
    except ImportError:
        svg_path = path if path.suffix.lower() == ".svg" else path.with_suffix(".svg")
        write_match_svg(
            svg_path,
            gt_segments=gt_segments,
            pred_segments=pred_segments,
            pred_to_gt=pred_to_gt,
            gt_to_pred=gt_to_pred,
            trajectory_xy=trajectory_xy,
            title=title,
        )
        return svg_path

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D

    matched_gt_ids = set(gt_to_pred.keys())
    matched_pred_ids = set(pred_to_gt.keys())
    gt_matched = [seg.points for seg in gt_segments if seg.sid in matched_gt_ids]
    gt_unmatched = [seg.points for seg in gt_segments if seg.sid not in matched_gt_ids]
    pred_matched = [seg.points for seg in pred_segments if seg.sid in matched_pred_ids]
    pred_unmatched = [seg.points for seg in pred_segments if seg.sid not in matched_pred_ids]

    fig, ax = plt.subplots(figsize=(13, 11), dpi=180)

    def add_collection(lines: list[np.ndarray], color: str, linewidth: float, alpha: float, zorder: int):
        if not lines:
            return
        ax.add_collection(
            LineCollection(lines, colors=color, linewidths=linewidth, alpha=alpha, zorder=zorder)
        )

    add_collection(gt_unmatched, "#9ca3af", 1.0, 0.55, 1)
    add_collection(gt_matched, "#2563eb", 1.25, 0.8, 2)
    add_collection(pred_matched, "#16a34a", 1.0, 0.9, 3)
    add_collection(pred_unmatched, "#dc2626", 1.4, 0.9, 4)

    if trajectory_xy is not None and len(trajectory_xy) >= 2:
        ax.plot(
            trajectory_xy[:, 0],
            trajectory_xy[:, 1],
            color="#111827",
            linewidth=0.8,
            alpha=0.45,
            zorder=0,
        )

    all_points = []
    for seg in gt_segments + pred_segments:
        if len(seg.points):
            all_points.append(seg.points)
    if trajectory_xy is not None and len(trajectory_xy):
        all_points.append(trajectory_xy)
    if all_points:
        pts = np.vstack(all_points)
        xmin, ymin = pts.min(axis=0)
        xmax, ymax = pts.max(axis=0)
        pad = max(max(xmax - xmin, ymax - ymin) * 0.04, 1.0)
        ax.set_xlim(xmin - pad, xmax + pad)
        ax.set_ylim(ymin - pad, ymax + pad)

    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xlabel("Map X (m)")
    ax.set_ylabel("Map Y (m)")
    ax.grid(True, color="#e5e7eb", linewidth=0.5)
    legend_items = [
        Line2D([0], [0], color="#2563eb", lw=2, label="GT matched"),
        Line2D([0], [0], color="#9ca3af", lw=2, label="GT unmatched"),
        Line2D([0], [0], color="#16a34a", lw=2, label="Pred matched"),
        Line2D([0], [0], color="#dc2626", lw=2, label="Pred unmatched"),
        Line2D([0], [0], color="#111827", lw=2, label="Reference trajectory"),
    ]
    ax.legend(handles=legend_items, loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def write_match_svg(
    path: Path,
    *,
    gt_segments: list[LocalSegment],
    pred_segments: list[LocalSegment],
    pred_to_gt: dict[int, int],
    gt_to_pred: dict[int, int],
    trajectory_xy: np.ndarray | None,
    title: str,
) -> None:
    from xml.sax.saxutils import escape

    matched_gt_ids = set(gt_to_pred.keys())
    matched_pred_ids = set(pred_to_gt.keys())
    groups = [
        ("GT unmatched", "#9ca3af", 1.0, 0.55, [seg.points for seg in gt_segments if seg.sid not in matched_gt_ids]),
        ("GT matched", "#2563eb", 1.25, 0.8, [seg.points for seg in gt_segments if seg.sid in matched_gt_ids]),
        ("Pred matched", "#16a34a", 1.0, 0.9, [seg.points for seg in pred_segments if seg.sid in matched_pred_ids]),
        ("Pred unmatched", "#dc2626", 1.4, 0.9, [seg.points for seg in pred_segments if seg.sid not in matched_pred_ids]),
    ]

    all_points = []
    for seg in gt_segments + pred_segments:
        if len(seg.points):
            all_points.append(seg.points)
    if trajectory_xy is not None and len(trajectory_xy):
        all_points.append(trajectory_xy)
    if not all_points:
        path.write_text("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"1200\" height=\"900\" />\n", encoding="utf-8")
        return

    pts = np.vstack(all_points)
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    pad = max(max(xmax - xmin, ymax - ymin) * 0.04, 1.0)
    xmin -= pad
    ymin -= pad
    xmax += pad
    ymax += pad

    width = 1400
    height = 1100
    margin = 70
    plot_w = width - 2 * margin
    plot_h = height - 2 * margin
    scale = min(plot_w / max(xmax - xmin, 1e-9), plot_h / max(ymax - ymin, 1e-9))
    draw_w = (xmax - xmin) * scale
    draw_h = (ymax - ymin) * scale
    x_offset = margin + (plot_w - draw_w) * 0.5
    y_offset = margin + (plot_h - draw_h) * 0.5

    def project(points: np.ndarray) -> str:
        coords = []
        for x, y in points:
            sx = x_offset + (float(x) - xmin) * scale
            sy = y_offset + draw_h - (float(y) - ymin) * scale
            coords.append(f"{sx:.2f},{sy:.2f}")
        return " ".join(coords)

    lines = [
        f"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\" viewBox=\"0 0 {width} {height}\">",
        "<rect width=\"100%\" height=\"100%\" fill=\"white\" />",
        f"<text x=\"{margin}\" y=\"36\" font-family=\"Arial, sans-serif\" font-size=\"22\" fill=\"#111827\">{escape(title)}</text>",
        f"<rect x=\"{x_offset:.2f}\" y=\"{y_offset:.2f}\" width=\"{draw_w:.2f}\" height=\"{draw_h:.2f}\" fill=\"none\" stroke=\"#e5e7eb\" stroke-width=\"1\" />",
    ]
    if trajectory_xy is not None and len(trajectory_xy) >= 2:
        lines.append(
            f"<polyline points=\"{project(trajectory_xy)}\" fill=\"none\" stroke=\"#111827\" stroke-width=\"0.8\" opacity=\"0.45\" />"
        )
    for _, color, linewidth, opacity, group_lines in groups:
        for points in group_lines:
            if len(points) >= 2:
                lines.append(
                    f"<polyline points=\"{project(points)}\" fill=\"none\" stroke=\"{color}\" stroke-width=\"{linewidth}\" opacity=\"{opacity}\" />"
                )

    legend_x = margin
    legend_y = height - margin + 12
    legend_items = [
        ("GT matched", "#2563eb"),
        ("GT unmatched", "#9ca3af"),
        ("Pred matched", "#16a34a"),
        ("Pred unmatched", "#dc2626"),
        ("Reference trajectory", "#111827"),
    ]
    for idx, (label, color) in enumerate(legend_items):
        x = legend_x + idx * 240
        lines.append(f"<line x1=\"{x}\" y1=\"{legend_y}\" x2=\"{x + 36}\" y2=\"{legend_y}\" stroke=\"{color}\" stroke-width=\"4\" />")
        lines.append(f"<text x=\"{x + 46}\" y=\"{legend_y + 5}\" font-family=\"Arial, sans-serif\" font-size=\"16\" fill=\"#111827\">{escape(label)}</text>")
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    require_gis_imports()
    import geopandas as gpd

    args = parse_args()
    gt_path = Path(args.gt)
    pred_path = Path(args.pred)
    gt_raw = gpd.read_file(gt_path)
    pred_raw = gpd.read_file(pred_path)
    metric_crs = choose_metric_crs(gt_raw, pred_raw)
    trajectory_xy = load_reference_trajectory(args, metric_crs)

    segment_lengths = sorted(set(float(v) for v in args.segment_lengths + [args.primary_segment_length]))
    apls_max_lengths = sorted(
        set(float(v) for v in (args.apls_max_lengths if args.apls_max_lengths else [args.apls_max_length]))
    )
    results = [
        evaluate_for_segment_length(
            args,
            metric_crs,
            gt_path,
            pred_path,
            segment_length,
            apls_max_lengths,
            trajectory_xy,
        )
        for segment_length in segment_lengths
    ]
    primary = min(results, key=lambda item: abs(item["segment_length_m"] - args.primary_segment_length))
    report = {
        "inputs": {
            "gt": str(gt_path),
            "pred": str(pred_path),
            "metric_crs": str(metric_crs),
            "lane_class": normalize_class(args.lane_class),
            "sample_spacing_m": args.sample_spacing,
            "segmentation_mode": args.segmentation_mode,
            "trajectory_tum": args.trajectory_tum,
            "trajectory_already_map_crs": args.trajectory_already_map_crs,
            "trajectory_lat_ref": args.lat_ref if args.trajectory_tum else None,
            "trajectory_lon_ref": args.lon_ref if args.trajectory_tum else None,
            "trajectory_h_ref": args.h_ref if args.trajectory_tum else None,
            "trajectory_normal_half_width_m": args.trajectory_normal_half_width,
            "trajectory_num_points": int(len(trajectory_xy)) if trajectory_xy is not None else None,
            "primary_segment_length_m": args.primary_segment_length,
            "segment_lengths_m": segment_lengths,
            "distance_threshold_m": args.distance_threshold,
            "endpoint_threshold_m": args.endpoint_threshold,
            "cross_chain_endpoint_threshold_m": args.cross_chain_endpoint_threshold,
            "chain_merge_threshold_m": args.chain_merge_threshold,
            "angle_threshold_deg": args.angle_threshold,
            "length_error_threshold": args.length_error_threshold,
            "apls_max_lengths_m": apls_max_lengths,
            "break_spacing_m": args.break_spacing,
            "duplicate_threshold_m": args.duplicate_threshold,
            "curvature_split_angle_deg": args.curvature_split_angle,
        },
        "primary": primary,
        "segment_length_results": results,
        "notes": {
            "local_graph": "Local lane segments are nodes; directed edges connect close end/start pairs with consistent heading.",
            "trajectory_normal_segmentation": "When enabled, GT and prediction lane chains are cut by the same station normals generated along the reference trajectory; long chains with too few normal intersections fall back to arclength cuts.",
            "matching": "Prediction and GT local segments are greedily one-to-one matched by Chamfer, endpoint errors, and heading difference.",
            "topology_recall": "Topology recall is path-based: a GT local edge is recovered by a directed predicted path with similar geometry and path length.",
            "direct_edge_recall": "Diagnostic recall requiring a matched GT local edge to be recovered by a single direct predicted edge.",
            "empty_intersections": "Areas with no GT lane-line segments are not sampled and are not penalized; intersection gaps without lane lines are ignored unless GT provides explicit connected lane-line edges.",
            "break_rate": "Fragmentation rate based on mean predicted local segment length relative to GT local segment length; lower is better.",
            "duplicate_rate": "Global near-overlapping parallel predicted duplicate length divided by total predicted length; lower is better.",
        },
    }

    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)
    output_plot = Path(args.output_plot) if args.output_plot else None
    report["outputs"] = {
        "json": str(output_json),
        "csv": str(output_csv),
        "plot": None,
    }
    if output_plot is not None:
        gt_plot_segments, pred_plot_segments, plot_pred_to_gt, plot_gt_to_pred = load_segments_for_plot(
            args,
            metric_crs,
            gt_path,
            pred_path,
            float(primary["segment_length_m"]),
            trajectory_xy,
        )
        output_plot = write_match_plot(
            output_plot,
            gt_segments=gt_plot_segments,
            pred_segments=pred_plot_segments,
            pred_to_gt=plot_pred_to_gt,
            gt_to_pred=plot_gt_to_pred,
            trajectory_xy=trajectory_xy,
            title="Topology matching, Lseg={:.1f}m".format(primary["segment_length_m"]),
        )
        report["outputs"]["plot"] = str(output_plot)

    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(output_csv, report)
    print(f"Wrote {output_json}")
    print(f"Wrote {output_csv}")
    if output_plot is not None:
        print(f"Wrote {output_plot}")
    print(
        "Primary Lseg={:.1f}m: topology_precision={:.4f}, topology_recall={:.4f}, topology_f1={:.4f}, "
        "break_rate={:.4f}, duplicate_rate={:.4f}, {}".format(
            primary["segment_length_m"],
            primary["local_topology_f1"]["precision"],
            primary["local_topology_f1"]["recall"],
            primary["local_topology_f1"]["f1"],
            primary["break_rate"]["break_rate"],
            primary["duplicate_rate"]["duplicate_rate"],
            ", ".join(
                f"apls@{key}={primary['local_directed_apls'][key]['score']:.4f}"
                for key in [apls_key(v) for v in apls_max_lengths]
            ),
        )
    )


if __name__ == "__main__":
    main()
