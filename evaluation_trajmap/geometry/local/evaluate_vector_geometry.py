#!/usr/bin/env python3
"""Evaluate vectorized HD-map geometry against a ground-truth shapefile.

The metric follows cd.txt:
  * sample each LineString into metric point sequences,
  * compute bidirectional Chamfer Distance (CD) between GT and prediction,
  * greedily match predictions to GT per class and radius,
  * report AP_c^r = TP / (TP + FP), plus recall/F1 for diagnosis,
  * aggregate hard/easy AP and mAP over classes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class Feature:
    fid: str
    cls: str
    lane_type: str
    points: np.ndarray
    length_m: float


@dataclass(frozen=True)
class ArrowFeature:
    fid: str
    polygon: Any
    center: np.ndarray
    area_m2: float


def require_gis_imports():
    try:
        import geopandas as gpd  # noqa: F401
        import shapely  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Missing GIS dependency. Run with an environment that has geopandas/shapely, "
            "for example:\n"
            "  conda run -n gotrackit python scripts/evaluate_vector_geometry.py ...\n"
        ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Geometry evaluation for vectorized HD-map shapefiles."
    )
    parser.add_argument("--gt", required=True, help="Ground-truth .shp path.")
    parser.add_argument("--pred", required=True, help="Predicted .shp path.")
    parser.add_argument(
        "--class-field",
        default="type",
        help="Attribute field used as map-element class. Default: type.",
    )
    parser.add_argument(
        "--gt-class-field",
        default=None,
        help="Override class field for GT if it differs from --class-field.",
    )
    parser.add_argument(
        "--pred-class-field",
        default=None,
        help="Override class field for prediction if it differs from --class-field.",
    )
    parser.add_argument(
        "--lane-type-field",
        default="lane_type",
        help="Attribute field used to split lane into dashed/solid. Default: lane_type.",
    )
    parser.add_argument(
        "--gt-lane-type-field",
        default=None,
        help="Override lane type field for GT if it differs from --lane-type-field.",
    )
    parser.add_argument(
        "--pred-lane-type-field",
        default=None,
        help="Override lane type field for prediction if it differs from --lane-type-field.",
    )
    parser.add_argument(
        "--classes",
        nargs="*",
        default=None,
        help="Classes to evaluate. Default: intersection of GT and prediction classes.",
    )
    parser.add_argument(
        "--ignore-class",
        action="store_true",
        help="Evaluate all line features as one class named 'all'.",
    )
    parser.add_argument(
        "--gt-class-map",
        default=None,
        help="Optional GT class mapping, e.g. '0=lane,1=arrow'. Applied after reading.",
    )
    parser.add_argument(
        "--pred-class-map",
        default=None,
        help="Optional prediction class mapping, e.g. 'None=unknown,lane=lane'.",
    )
    parser.add_argument(
        "--lane-class",
        default="lane",
        help="Class value treated as lane-line class. Default: lane.",
    )
    parser.add_argument(
        "--arrow-class",
        default="arrow",
        help="Class value treated as arrow class. Default: arrow.",
    )
    parser.add_argument(
        "--dashed-values",
        nargs="+",
        default=["dashed"],
        help="lane_type values treated as dashed lanes.",
    )
    parser.add_argument(
        "--solid-values",
        nargs="+",
        default=["solid", "left solid", "right solid"],
        help="lane_type values treated as solid lanes.",
    )
    parser.add_argument(
        "--sample-spacing",
        type=float,
        default=0.2,
        help="Point sampling spacing on each line, in meters. Default: 0.2.",
    )
    parser.add_argument(
        "--hard-radii",
        nargs="+",
        type=float,
        default=[0.2, 0.5, 1.0],
        help="Hard setting tolerance radii in meters.",
    )
    parser.add_argument(
        "--easy-radii",
        nargs="+",
        type=float,
        default=[0.5, 1.0, 1.5],
        help="Easy setting tolerance radii in meters.",
    )
    parser.add_argument(
        "--arrow-iou-thresholds",
        nargs="+",
        type=float,
        default=[0.1, 0.25, 0.5],
        help="Rotated IoU thresholds for arrow AP/precision.",
    )
    parser.add_argument(
        "--arrow-center-thresholds",
        nargs="+",
        type=float,
        default=[1.0, 2.0, 5.0],
        help="Center-distance thresholds in meters for arrow AP/precision.",
    )
    parser.add_argument(
        "--arrow-min-area",
        type=float,
        default=0.5,
        help="Minimum polygon area in square meters kept for arrow evaluation.",
    )
    parser.add_argument(
        "--max-candidate-distance",
        type=float,
        default=None,
        help=(
            "Optional candidate-pair prefilter in meters. Default uses "
            "max(all radii) + 2 * sample spacing."
        ),
    )
    parser.add_argument(
        "--trajectory-tum",
        default=None,
        help="Optional TUM odometry trajectory txt. Used only for report metadata.",
    )
    parser.add_argument(
        "--output-json",
        default="geometry_eval_results.json",
        help="Output JSON report path.",
    )
    parser.add_argument(
        "--output-csv",
        default="geometry_eval_results.csv",
        help="Output per-radius CSV path.",
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
    geom_type = geom.geom_type
    if geom_type == "LineString":
        yield geom
    elif geom_type == "MultiLineString":
        yield from geom.geoms
    elif geom_type == "GeometryCollection":
        for part in geom.geoms:
            yield from iter_line_parts(part)


def iter_polygon_parts(geom) -> Iterable[Any]:
    if geom is None or geom.is_empty:
        return
    if geom.geom_type == "Polygon":
        yield geom
    elif geom.geom_type == "MultiPolygon":
        yield from geom.geoms
    elif geom.geom_type == "GeometryCollection":
        for part in geom.geoms:
            yield from iter_polygon_parts(part)


def sample_line(line, spacing: float) -> np.ndarray:
    length = float(line.length)
    if length == 0.0:
        coords = np.asarray(line.coords, dtype=float)
        return coords[:1, :2]
    n_steps = max(int(math.ceil(length / spacing)), 1)
    distances = np.linspace(0.0, length, n_steps + 1)
    points = [line.interpolate(float(d)) for d in distances]
    return np.asarray([(p.x, p.y) for p in points], dtype=float)


def load_features(
    shp_path: Path,
    class_field: str,
    metric_crs,
    spacing: float,
    *,
    ignore_class: bool = False,
    class_map: dict[str, str] | None = None,
    lane_type_field: str | None = None,
) -> list[Feature]:
    import geopandas as gpd

    gdf = gpd.read_file(shp_path)
    if class_field not in gdf.columns:
        raise ValueError(
            f"Class field '{class_field}' not found in {shp_path}. "
            f"Available fields: {list(gdf.columns)}"
        )
    if lane_type_field and lane_type_field not in gdf.columns:
        lane_type_field = None
    if gdf.crs is None:
        raise ValueError(f"{shp_path} has no CRS; cannot evaluate metric distances.")
    gdf = gdf.to_crs(metric_crs)

    features: list[Feature] = []
    class_map = class_map or {}
    for row_idx, row in gdf.iterrows():
        cls = "all" if ignore_class else normalize_class(row[class_field])
        cls = class_map.get(cls, cls)
        lane_type = normalize_class(row[lane_type_field]) if lane_type_field else "unknown"
        for part_idx, line in enumerate(iter_line_parts(row.geometry)):
            pts = sample_line(line, spacing)
            if len(pts) == 0:
                continue
            features.append(
                Feature(
                    fid=f"{row_idx}:{part_idx}",
                    cls=cls,
                    lane_type=lane_type,
                    points=pts,
                    length_m=float(line.length),
                )
            )
    return features


def linework_to_polygon(geom):
    """Convert arrow rectangle linework to a polygon region for rotated IoU."""
    from shapely.geometry import Polygon
    from shapely.ops import polygonize, unary_union

    polygons = list(iter_polygon_parts(geom))
    if polygons:
        return max(polygons, key=lambda poly: poly.area)

    line_parts = list(iter_line_parts(geom))
    closed_polygons = []
    for line in line_parts:
        coords = list(line.coords)
        if len(coords) >= 4:
            first = np.asarray(coords[0][:2], dtype=float)
            last = np.asarray(coords[-1][:2], dtype=float)
            if np.linalg.norm(first - last) <= 0.05:
                try:
                    poly = Polygon(coords)
                    if poly.is_valid and poly.area > 0:
                        closed_polygons.append(poly)
                except Exception:
                    pass
    if closed_polygons:
        return max(closed_polygons, key=lambda poly: poly.area)

    if line_parts:
        merged = unary_union(line_parts)
        polygonized = [poly for poly in polygonize(merged) if poly.is_valid and poly.area > 0]
        if polygonized:
            return max(polygonized, key=lambda poly: poly.area)

    # Fallback for slightly open rectangle linework: evaluate its minimum rotated rectangle.
    rect = geom.minimum_rotated_rectangle
    if rect.is_valid and rect.area > 0:
        return rect
    return None


def load_arrow_features(
    shp_path: Path,
    class_field: str,
    metric_crs,
    arrow_class: str,
    *,
    class_map: dict[str, str] | None = None,
    min_area: float = 0.5,
) -> list[ArrowFeature]:
    import geopandas as gpd
    from shapely.ops import polygonize, unary_union

    gdf = gpd.read_file(shp_path)
    if class_field not in gdf.columns:
        raise ValueError(
            f"Class field '{class_field}' not found in {shp_path}. "
            f"Available fields: {list(gdf.columns)}"
        )
    if gdf.crs is None:
        raise ValueError(f"{shp_path} has no CRS; cannot evaluate metric distances.")
    gdf = gdf.to_crs(metric_crs)
    class_map = class_map or {}
    arrow_geoms = []
    fallback_polygons = []
    arrow_row_ids = []
    for row_idx, row in gdf.iterrows():
        cls = class_map.get(normalize_class(row[class_field]), normalize_class(row[class_field]))
        if cls != arrow_class:
            continue
        arrow_geoms.append(row.geometry)
        arrow_row_ids.append(row_idx)
        polygon = linework_to_polygon(row.geometry)
        if polygon is not None and not polygon.is_empty and polygon.area >= min_area:
            fallback_polygons.append((str(row_idx), polygon))

    polygons = []
    if arrow_geoms:
        merged = unary_union(arrow_geoms)
        polygons = [
            polygon
            for polygon in polygonize(merged)
            if polygon.is_valid and polygon.area >= min_area
        ]
    if polygons:
        arrow_features = []
        for idx, polygon in enumerate(polygons):
            center = polygon.centroid
            arrow_features.append(
                ArrowFeature(
                    fid=f"polygonized:{idx}",
                    polygon=polygon,
                    center=np.asarray([center.x, center.y], dtype=float),
                    area_m2=float(polygon.area),
                )
            )
        return arrow_features

    arrow_features: list[ArrowFeature] = []
    for fid, polygon in fallback_polygons:
        center = polygon.centroid
        arrow_features.append(
            ArrowFeature(
                fid=fid,
                polygon=polygon,
                center=np.asarray([center.x, center.y], dtype=float),
                area_m2=float(polygon.area),
            )
        )
    return arrow_features


def chamfer_distance(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return float("inf")
    tree_b = cKDTree(b)
    tree_a = cKDTree(a)
    d_ab = tree_b.query(a, k=1, workers=-1)[0].mean()
    d_ba = tree_a.query(b, k=1, workers=-1)[0].mean()
    return float(d_ab + d_ba)


def chamfer_components(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    if len(a) == 0 or len(b) == 0:
        return {
            "gt_to_pred_m": float("inf"),
            "pred_to_gt_m": float("inf"),
            "symmetric_cd_m": float("inf"),
        }
    tree_b = cKDTree(b)
    tree_a = cKDTree(a)
    gt_to_pred = float(tree_b.query(a, k=1, workers=-1)[0].mean())
    pred_to_gt = float(tree_a.query(b, k=1, workers=-1)[0].mean())
    return {
        "gt_to_pred_m": gt_to_pred,
        "pred_to_gt_m": pred_to_gt,
        "symmetric_cd_m": gt_to_pred + pred_to_gt,
    }


def centroid(points: np.ndarray) -> np.ndarray:
    return points.mean(axis=0)


def cd_pairs(
    gt_features: list[Feature],
    pred_features: list[Feature],
    candidate_distance: float,
) -> list[tuple[float, int, int]]:
    if not gt_features or not pred_features:
        return []
    gt_centers = np.asarray([centroid(f.points) for f in gt_features])
    pred_centers = np.asarray([centroid(f.points) for f in pred_features])
    gt_radii = np.asarray(
        [np.linalg.norm(f.points - gt_centers[i], axis=1).max() for i, f in enumerate(gt_features)]
    )
    pred_radii = np.asarray(
        [
            np.linalg.norm(f.points - pred_centers[i], axis=1).max()
            for i, f in enumerate(pred_features)
        ]
    )

    # Centroid distance lower-bounds possible geometry distance only loosely.
    # Add each feature's point spread so long lines are not accidentally filtered.
    tree = cKDTree(pred_centers)
    max_pred_radius = float(pred_radii.max(initial=0.0))
    pairs: list[tuple[float, int, int]] = []
    for gi, gt in enumerate(gt_features):
        search_radius = candidate_distance + float(gt_radii[gi]) + max_pred_radius
        for pi in tree.query_ball_point(gt_centers[gi], r=search_radius):
            center_dist = float(np.linalg.norm(gt_centers[gi] - pred_centers[pi]))
            if center_dist > candidate_distance + gt_radii[gi] + pred_radii[pi]:
                continue
            cd = chamfer_distance(gt.points, pred_features[pi].points)
            pairs.append((cd, gi, pi))
    pairs.sort(key=lambda item: item[0])
    return pairs


def match_at_radius(
    sorted_pairs: list[tuple[float, int, int]], n_gt: int, n_pred: int, radius: float
) -> dict[str, Any]:
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    matched_cds: list[float] = []
    for cd, gi, pi in sorted_pairs:
        if cd >= radius:
            break
        if gi in matched_gt or pi in matched_pred:
            continue
        matched_gt.add(gi)
        matched_pred.add(pi)
        matched_cds.append(cd)

    tp = len(matched_cds)
    fp = n_pred - tp
    fn = n_gt - tp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "radius_m": radius,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "ap_precision": precision,
        "recall": recall,
        "f1": f1,
        "matched_mean_cd_m": float(np.mean(matched_cds)) if matched_cds else None,
        "matched_median_cd_m": float(np.median(matched_cds)) if matched_cds else None,
    }


def best_chamfer_pair_stats(sorted_pairs: list[tuple[float, int, int]], n_limit: int = 100):
    cds = [pair[0] for pair in sorted_pairs[:n_limit]]
    if not cds:
        return {"best_pair_cd_m": None, f"first_{n_limit}_pair_mean_cd_m": None}
    return {
        "best_pair_cd_m": float(cds[0]),
        f"first_{n_limit}_pair_mean_cd_m": float(np.mean(cds)),
    }


def point_cloud_metrics(
    gt_features: list[Feature], pred_features: list[Feature], radii: list[float]
) -> dict[str, Any]:
    """Distance statistics that ignore feature identity and measure visual overlap."""
    if not gt_features or not pred_features:
        return {}
    gt_points = np.vstack([feature.points for feature in gt_features])
    pred_points = np.vstack([feature.points for feature in pred_features])
    gt_tree = cKDTree(gt_points)
    pred_tree = cKDTree(pred_points)
    gt_to_pred = pred_tree.query(gt_points, k=1, workers=-1)[0]
    pred_to_gt = gt_tree.query(pred_points, k=1, workers=-1)[0]

    def summarize(distances: np.ndarray) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "mean_m": float(distances.mean()),
            "median_m": float(np.median(distances)),
            "p80_m": float(np.quantile(distances, 0.80)),
            "p90_m": float(np.quantile(distances, 0.90)),
            "p95_m": float(np.quantile(distances, 0.95)),
        }
        for radius in radii:
            payload[f"ratio_within_{radius:g}m"] = float((distances <= radius).mean())
        return payload

    return {
        "note": "Point-cloud nearest-neighbor overlap; ignores one-to-one feature matching.",
        "gt_to_pred": summarize(gt_to_pred),
        "pred_to_gt": summarize(pred_to_gt),
        "symmetric_mean_m": float(gt_to_pred.mean() + pred_to_gt.mean()),
    }


def global_chamfer_metrics(
    gt_features: list[Feature], pred_features: list[Feature], radii: list[float]
) -> dict[str, Any]:
    if not gt_features or not pred_features:
        return {}
    gt_points = np.vstack([feature.points for feature in gt_features])
    pred_points = np.vstack([feature.points for feature in pred_features])
    tree_pred = cKDTree(pred_points)
    tree_gt = cKDTree(gt_points)
    gt_to_pred = tree_pred.query(gt_points, k=1, workers=-1)[0]
    pred_to_gt = tree_gt.query(pred_points, k=1, workers=-1)[0]

    def direction_payload(distances: np.ndarray) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "mean_m": float(distances.mean()),
            "median_m": float(np.median(distances)),
            "p80_m": float(np.quantile(distances, 0.80)),
            "p90_m": float(np.quantile(distances, 0.90)),
            "p95_m": float(np.quantile(distances, 0.95)),
        }
        for radius in radii:
            payload[f"coverage_within_{radius:g}m"] = float((distances <= radius).mean())
        return payload

    return {
        "metric": "global_bidirectional_chamfer_distance",
        "num_gt_sample_points": int(len(gt_points)),
        "num_pred_sample_points": int(len(pred_points)),
        "gt_to_pred": direction_payload(gt_to_pred),
        "pred_to_gt": direction_payload(pred_to_gt),
        "symmetric_cd_m": float(gt_to_pred.mean() + pred_to_gt.mean()),
        "note": (
            "This is the primary lane geometry metric: all lane-line samples are "
            "evaluated as two point sets, so GT/prediction point counts may differ."
        ),
    }


def evaluate_lane_geometry(
    gt_features: list[Feature],
    pred_features: list[Feature],
    radii: list[float],
    candidate_distance: float,
    *,
    label: str = "lane",
) -> dict[str, Any]:
    pairs = cd_pairs(gt_features, pred_features, candidate_distance)
    radius_metrics = [
        match_at_radius(pairs, len(gt_features), len(pred_features), radius) for radius in radii
    ]
    return {
        "label": label,
        "metric": "bidirectional_chamfer_distance",
        "note": (
            "Each direction is a nearest-neighbor mean, so GT and prediction may have "
            "different sampled point counts. symmetric_cd = gt_to_pred + pred_to_gt."
        ),
        "num_gt": len(gt_features),
        "num_pred": len(pred_features),
        "gt_total_length_m": float(sum(f.length_m for f in gt_features)),
        "pred_total_length_m": float(sum(f.length_m for f in pred_features)),
        "global_chamfer": global_chamfer_metrics(gt_features, pred_features, radii),
        "candidate_pairs": len(pairs),
        **best_chamfer_pair_stats(pairs),
        "point_cloud_overlap": point_cloud_metrics(gt_features, pred_features, radii),
        "radii": radius_metrics,
    }


def summarize_setting(items: list[dict[str, Any]], key: str, threshold_key: str, thresholds: list[float]):
    by_threshold = {round(item[threshold_key], 6): item for item in items}
    values = [
        by_threshold[round(threshold, 6)][key]
        for threshold in thresholds
        if round(threshold, 6) in by_threshold
    ]
    return float(np.mean(values)) if values else 0.0


def rotated_iou(poly_a, poly_b) -> float:
    if poly_a is None or poly_b is None or poly_a.is_empty or poly_b.is_empty:
        return 0.0
    if not poly_a.is_valid:
        poly_a = poly_a.buffer(0)
    if not poly_b.is_valid:
        poly_b = poly_b.buffer(0)
    union_area = poly_a.union(poly_b).area
    if union_area <= 0:
        return 0.0
    return float(poly_a.intersection(poly_b).area / union_area)


def arrow_pairs(
    gt_arrows: list[ArrowFeature],
    pred_arrows: list[ArrowFeature],
    candidate_center_distance: float,
) -> list[tuple[float, float, int, int]]:
    if not gt_arrows or not pred_arrows:
        return []
    pred_centers = np.asarray([feature.center for feature in pred_arrows])
    tree = cKDTree(pred_centers)
    pairs: list[tuple[float, float, int, int]] = []
    for gi, gt in enumerate(gt_arrows):
        for pi in tree.query_ball_point(gt.center, r=candidate_center_distance):
            center_error = float(np.linalg.norm(gt.center - pred_arrows[pi].center))
            iou = rotated_iou(gt.polygon, pred_arrows[pi].polygon)
            pairs.append((iou, center_error, gi, pi))
    pairs.sort(key=lambda item: (-item[0], item[1]))
    return pairs


def match_arrow_iou(
    sorted_pairs: list[tuple[float, float, int, int]], n_gt: int, n_pred: int, iou_threshold: float
) -> dict[str, Any]:
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    matched_iou: list[float] = []
    matched_center: list[float] = []
    for iou, center_error, gi, pi in sorted_pairs:
        if iou < iou_threshold:
            continue
        if gi in matched_gt or pi in matched_pred:
            continue
        matched_gt.add(gi)
        matched_pred.add(pi)
        matched_iou.append(iou)
        matched_center.append(center_error)

    tp = len(matched_iou)
    fp = n_pred - tp
    fn = n_gt - tp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "iou_threshold": iou_threshold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "ap_precision": precision,
        "recall": recall,
        "f1": f1,
        "matched_mean_iou": float(np.mean(matched_iou)) if matched_iou else None,
        "matched_median_iou": float(np.median(matched_iou)) if matched_iou else None,
        "matched_mean_center_error_m": float(np.mean(matched_center)) if matched_center else None,
        "matched_median_center_error_m": float(np.median(matched_center)) if matched_center else None,
    }


def match_arrow_center(
    sorted_pairs: list[tuple[float, float, int, int]],
    n_gt: int,
    n_pred: int,
    center_threshold: float,
) -> dict[str, Any]:
    pairs = sorted(sorted_pairs, key=lambda item: (item[1], -item[0]))
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    matched_iou: list[float] = []
    matched_center: list[float] = []
    for iou, center_error, gi, pi in pairs:
        if center_error > center_threshold:
            break
        if gi in matched_gt or pi in matched_pred:
            continue
        matched_gt.add(gi)
        matched_pred.add(pi)
        matched_iou.append(iou)
        matched_center.append(center_error)
    tp = len(matched_center)
    fp = n_pred - tp
    fn = n_gt - tp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "center_threshold_m": center_threshold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "ap_precision": precision,
        "recall": recall,
        "f1": f1,
        "matched_mean_iou": float(np.mean(matched_iou)) if matched_iou else None,
        "matched_median_iou": float(np.median(matched_iou)) if matched_iou else None,
        "matched_mean_center_error_m": float(np.mean(matched_center)) if matched_center else None,
        "matched_median_center_error_m": float(np.median(matched_center)) if matched_center else None,
    }


def best_arrow_match_summary(sorted_pairs: list[tuple[float, float, int, int]], n_gt: int, n_pred: int):
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    ious: list[float] = []
    centers: list[float] = []
    for iou, center_error, gi, pi in sorted_pairs:
        if gi in matched_gt or pi in matched_pred:
            continue
        matched_gt.add(gi)
        matched_pred.add(pi)
        ious.append(iou)
        centers.append(center_error)
    return {
        "matched_pairs_without_threshold": len(ious),
        "unmatched_gt_without_threshold": n_gt - len(ious),
        "unmatched_pred_without_threshold": n_pred - len(ious),
        "mean_iou_without_threshold": float(np.mean(ious)) if ious else None,
        "median_iou_without_threshold": float(np.median(ious)) if ious else None,
        "mean_center_error_m_without_threshold": float(np.mean(centers)) if centers else None,
        "median_center_error_m_without_threshold": float(np.median(centers)) if centers else None,
    }


def evaluate_arrow_geometry(
    gt_arrows: list[ArrowFeature],
    pred_arrows: list[ArrowFeature],
    iou_thresholds: list[float],
    center_thresholds: list[float],
) -> dict[str, Any]:
    max_center_threshold = max(center_thresholds) if center_thresholds else 5.0
    candidate_center_distance = max(50.0, max_center_threshold * 3.0)
    pairs = arrow_pairs(gt_arrows, pred_arrows, candidate_center_distance)
    iou_metrics = [
        match_arrow_iou(pairs, len(gt_arrows), len(pred_arrows), threshold)
        for threshold in iou_thresholds
    ]
    center_metrics = [
        match_arrow_center(pairs, len(gt_arrows), len(pred_arrows), threshold)
        for threshold in center_thresholds
    ]
    return {
        "metric": "rotated_iou_and_center_translation_error",
        "note": (
            "Arrow linework is converted to polygon regions. Closed rectangles are "
            "polygonized; open rectangles fall back to minimum rotated rectangles."
        ),
        "num_gt": len(gt_arrows),
        "num_pred": len(pred_arrows),
        "gt_total_area_m2": float(sum(item.area_m2 for item in gt_arrows)),
        "pred_total_area_m2": float(sum(item.area_m2 for item in pred_arrows)),
        "candidate_pairs": len(pairs),
        **best_arrow_match_summary(pairs, len(gt_arrows), len(pred_arrows)),
        "iou_thresholds": iou_metrics,
        "center_thresholds": center_metrics,
        "mean_ap_iou": summarize_setting(
            iou_metrics, "ap_precision", "iou_threshold", iou_thresholds
        ),
        "mean_ap_center": summarize_setting(
            center_metrics, "ap_precision", "center_threshold_m", center_thresholds
        ),
    }


def read_tum_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    meta: dict[str, Any] = {
        "path": str(path),
        "exists": True,
        "rows": int(data.shape[0]),
        "columns": int(data.shape[1]),
    }
    if data.shape[1] >= 4 and data.shape[0] >= 2:
        xyz = data[:, 1:4]
        diffs = np.diff(xyz, axis=0)
        meta["local_xyz_path_length_m"] = float(np.linalg.norm(diffs, axis=1).sum())
        meta["local_xyz_bbox_min"] = xyz.min(axis=0).tolist()
        meta["local_xyz_bbox_max"] = xyz.max(axis=0).tolist()
        meta["time_start"] = float(data[0, 0])
        meta["time_end"] = float(data[-1, 0])
    return meta


def write_csv(path: Path, report: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    for lane_key in ["lane_overall", "lane_dashed", "lane_solid"]:
        lane = report.get(lane_key, {})
        for item in lane.get("radii", []):
            rows.append(
                {
                    "category": lane_key,
                    "metric_family": "chamfer_distance",
                    "threshold_type": "radius_m",
                    "threshold": item["radius_m"],
                    "num_gt": lane.get("num_gt"),
                    "num_pred": lane.get("num_pred"),
                    "tp": item["tp"],
                    "fp": item["fp"],
                    "fn": item["fn"],
                    "ap_precision": item["ap_precision"],
                    "recall": item["recall"],
                    "f1": item["f1"],
                    "matched_mean_cd_m": item["matched_mean_cd_m"],
                    "matched_median_cd_m": item["matched_median_cd_m"],
                    "matched_mean_iou": None,
                    "matched_median_iou": None,
                    "matched_mean_center_error_m": None,
                    "matched_median_center_error_m": None,
                }
            )
    arrow = report.get("arrow", {})
    for item in arrow.get("iou_thresholds", []):
        rows.append(
            {
                "category": "arrow",
                "metric_family": "rotated_iou",
                "threshold_type": "iou_threshold",
                "threshold": item["iou_threshold"],
                "num_gt": arrow.get("num_gt"),
                "num_pred": arrow.get("num_pred"),
                "tp": item["tp"],
                "fp": item["fp"],
                "fn": item["fn"],
                "ap_precision": item["ap_precision"],
                "recall": item["recall"],
                "f1": item["f1"],
                "matched_mean_cd_m": None,
                "matched_median_cd_m": None,
                "matched_mean_iou": item["matched_mean_iou"],
                "matched_median_iou": item["matched_median_iou"],
                "matched_mean_center_error_m": item["matched_mean_center_error_m"],
                "matched_median_center_error_m": item["matched_median_center_error_m"],
            }
        )
    for item in arrow.get("center_thresholds", []):
        rows.append(
            {
                "category": "arrow",
                "metric_family": "center_translation_error",
                "threshold_type": "center_threshold_m",
                "threshold": item["center_threshold_m"],
                "num_gt": arrow.get("num_gt"),
                "num_pred": arrow.get("num_pred"),
                "tp": item["tp"],
                "fp": item["fp"],
                "fn": item["fn"],
                "ap_precision": item["ap_precision"],
                "recall": item["recall"],
                "f1": item["f1"],
                "matched_mean_cd_m": None,
                "matched_median_cd_m": None,
                "matched_mean_iou": item["matched_mean_iou"],
                "matched_median_iou": item["matched_median_iou"],
                "matched_mean_center_error_m": item["matched_mean_center_error_m"],
                "matched_median_center_error_m": item["matched_median_center_error_m"],
            }
        )
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    require_gis_imports()
    import geopandas as gpd

    args = parse_args()
    gt_path = Path(args.gt)
    pred_path = Path(args.pred)
    gt_field = args.gt_class_field or args.class_field
    pred_field = args.pred_class_field or args.class_field
    gt_lane_type_field = args.gt_lane_type_field or args.lane_type_field
    pred_lane_type_field = args.pred_lane_type_field or args.lane_type_field
    gt_class_map = parse_class_map(args.gt_class_map)
    pred_class_map = parse_class_map(args.pred_class_map)

    gt_raw = gpd.read_file(gt_path)
    pred_raw = gpd.read_file(pred_path)
    metric_crs = choose_metric_crs(gt_raw, pred_raw)
    lane_class = normalize_class(args.lane_class)
    arrow_class = normalize_class(args.arrow_class)
    gt_features = load_features(
        gt_path,
        gt_field,
        metric_crs,
        args.sample_spacing,
        ignore_class=args.ignore_class,
        class_map=gt_class_map,
        lane_type_field=gt_lane_type_field,
    )
    pred_features = load_features(
        pred_path,
        pred_field,
        metric_crs,
        args.sample_spacing,
        ignore_class=args.ignore_class,
        class_map=pred_class_map,
        lane_type_field=pred_lane_type_field,
    )
    gt_lane_features = [feature for feature in gt_features if feature.cls == lane_class]
    pred_lane_features = [feature for feature in pred_features if feature.cls == lane_class]
    dashed_values = {normalize_class(value) for value in args.dashed_values}
    solid_values = {normalize_class(value) for value in args.solid_values}
    gt_dashed_features = [
        feature for feature in gt_lane_features if feature.lane_type in dashed_values
    ]
    pred_dashed_features = [
        feature for feature in pred_lane_features if feature.lane_type in dashed_values
    ]
    gt_solid_features = [
        feature for feature in gt_lane_features if feature.lane_type in solid_values
    ]
    pred_solid_features = [
        feature for feature in pred_lane_features if feature.lane_type in solid_values
    ]
    gt_arrow_features = load_arrow_features(
        gt_path,
        gt_field,
        metric_crs,
        arrow_class,
        class_map=gt_class_map,
        min_area=args.arrow_min_area,
    )
    pred_arrow_features = load_arrow_features(
        pred_path,
        pred_field,
        metric_crs,
        arrow_class,
        class_map=pred_class_map,
        min_area=args.arrow_min_area,
    )

    gt_classes = {f.cls for f in gt_features}
    pred_classes = {f.cls for f in pred_features}
    missing = []
    if not gt_lane_features:
        missing.append(f"GT has no lane class '{lane_class}'")
    if not pred_lane_features:
        missing.append(f"Prediction has no lane class '{lane_class}'")
    if not gt_arrow_features:
        missing.append(f"GT has no arrow class '{arrow_class}'")
    if not pred_arrow_features:
        missing.append(f"Prediction has no arrow class '{arrow_class}'")
    if missing:
        print("warning: " + "; ".join(missing))

    all_radii = sorted(set(args.hard_radii + args.easy_radii))
    candidate_distance = args.max_candidate_distance
    if candidate_distance is None:
        candidate_distance = max(all_radii) + 2.0 * args.sample_spacing

    lane_report = evaluate_lane_geometry(
        gt_lane_features,
        pred_lane_features,
        all_radii,
        candidate_distance,
        label="lane_overall",
    )
    lane_dashed_report = evaluate_lane_geometry(
        gt_dashed_features,
        pred_dashed_features,
        all_radii,
        candidate_distance,
        label="lane_dashed",
    )
    lane_solid_report = evaluate_lane_geometry(
        gt_solid_features,
        pred_solid_features,
        all_radii,
        candidate_distance,
        label="lane_solid",
    )
    arrow_report = evaluate_arrow_geometry(
        gt_arrow_features,
        pred_arrow_features,
        args.arrow_iou_thresholds,
        args.arrow_center_thresholds,
    )
    report = {
        "inputs": {
            "gt": str(gt_path),
            "pred": str(pred_path),
            "gt_class_field": gt_field,
            "pred_class_field": pred_field,
            "metric_crs": str(metric_crs),
            "sample_spacing_m": args.sample_spacing,
            "candidate_distance_m": candidate_distance,
            "ignore_class": args.ignore_class,
            "gt_class_map": gt_class_map,
            "pred_class_map": pred_class_map,
            "gt_classes": sorted(gt_classes),
            "pred_classes": sorted(pred_classes),
            "lane_class": lane_class,
            "arrow_class": arrow_class,
            "gt_lane_type_field": gt_lane_type_field,
            "pred_lane_type_field": pred_lane_type_field,
            "dashed_values": sorted(dashed_values),
            "solid_values": sorted(solid_values),
            "arrow_min_area_m2": args.arrow_min_area,
        },
        "lane_overall": lane_report,
        "lane_dashed": lane_dashed_report,
        "lane_solid": lane_solid_report,
        "arrow": arrow_report,
        "summary": {
            "radii_m": args.hard_radii,
            "lane_overall_hard_ap": summarize_setting(
                lane_report["radii"], "ap_precision", "radius_m", args.hard_radii
            ),
            "lane_overall_easy_ap": summarize_setting(
                lane_report["radii"], "ap_precision", "radius_m", args.easy_radii
            ),
            "lane_dashed_hard_ap": summarize_setting(
                lane_dashed_report["radii"], "ap_precision", "radius_m", args.hard_radii
            ),
            "lane_dashed_easy_ap": summarize_setting(
                lane_dashed_report["radii"], "ap_precision", "radius_m", args.easy_radii
            ),
            "lane_solid_hard_ap": summarize_setting(
                lane_solid_report["radii"], "ap_precision", "radius_m", args.hard_radii
            ),
            "lane_solid_easy_ap": summarize_setting(
                lane_solid_report["radii"], "ap_precision", "radius_m", args.easy_radii
            ),
            "arrow_mean_ap_iou": arrow_report["mean_ap_iou"],
            "arrow_mean_ap_center": arrow_report["mean_ap_center"],
        },
    }
    if args.trajectory_tum:
        report["trajectory_tum"] = read_tum_metadata(Path(args.trajectory_tum))

    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(output_csv, report)

    print(f"metric_crs: {metric_crs}")
    print(f"lane overall: gt={lane_report['num_gt']} pred={lane_report['num_pred']}")
    print(
        f"lane dashed: gt={lane_dashed_report['num_gt']} pred={lane_dashed_report['num_pred']}"
    )
    print(f"lane solid: gt={lane_solid_report['num_gt']} pred={lane_solid_report['num_pred']}")
    print(f"arrow: gt={arrow_report['num_gt']} pred={arrow_report['num_pred']}")
    print(f"lane overall hard AP: {report['summary']['lane_overall_hard_ap']:.6f}")
    print(f"lane overall easy AP: {report['summary']['lane_overall_easy_ap']:.6f}")
    print(f"lane dashed hard AP: {report['summary']['lane_dashed_hard_ap']:.6f}")
    print(f"lane solid hard AP: {report['summary']['lane_solid_hard_ap']:.6f}")
    print(f"arrow mean AP IoU: {report['summary']['arrow_mean_ap_iou']:.6f}")
    print(f"arrow mean AP center: {report['summary']['arrow_mean_ap_center']:.6f}")
    print(f"wrote: {output_json}")
    print(f"wrote: {output_csv}")


if __name__ == "__main__":
    main()
