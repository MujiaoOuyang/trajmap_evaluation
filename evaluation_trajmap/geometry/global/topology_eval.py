import os

os.environ.setdefault("SHAPE_RESTORE_SHX", "NO")

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import networkx as nx
import numpy as np
from matplotlib.lines import Line2D
from scipy.optimize import linear_sum_assignment
from scipy.spatial import KDTree
from shapely.geometry import LineString, Point


TARGET_CRS = "EPSG:32650"
GT_SHP_PATH = "/home/oymj/evaluation_trajmap/topology/groundtruth/groundtruth_topo_all.shp"
GT_WITH_NODES_PATH = "/home/oymj/tramap/shp_transform/topology_eval_outputs_evaltrajmap/groundtruth_topo_all_with_nodes.shp"
PRED_SHP_PATH = "/home/oymj/evaluation_trajmap/topology/my/mytopo_all.shp"
PRED_WITH_NODES_PATH = "/home/oymj/tramap/shp_transform/topology_eval_outputs_evaltrajmap/mytopo_all_with_nodes.shp"
OUTPUT_DIR = Path("/home/oymj/tramap/shp_transform/topology_eval_outputs_evaltrajmap")
DEBUG_DIR = OUTPUT_DIR / "debug_reports"
MATCH_TOLERANCE = 14.5
MATCH_SEARCH_RADIUS = 45.0
MATCH_NEAREST_CANDIDATES = 8
ANGLE_TOLERANCE_DEG = 18.0
GT_NODE_MERGE_TOLERANCE = 0.1
GT_INTERSECTION_BUFFER = 0.0
FIG_DPI = 600
FIGSIZE_SQUARE = (11.2, 11.2)
LEGEND_FONTSIZE = 8.5
TITLE_FONTSIZE = 13
LABEL_PADDING_RATIO = 0.04


def read_metric_gdf(shp_path, target_crs=TARGET_CRS):
    gdf = gpd.read_file(shp_path)
    if gdf.empty:
        raise ValueError(f"输入文件为空: {shp_path}")

    if gdf.crs is None:
        bounds = gdf.total_bounds
        if bounds[0] > -180 and bounds[2] < 180 and bounds[1] > -90 and bounds[3] < 90:
            gdf = gdf.set_crs("EPSG:4326")

    if target_crs is not None and gdf.crs is not None and str(gdf.crs) != target_crs:
        gdf = gdf.to_crs(target_crs)
    return gdf


def normalize_text(value):
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"", "none", "nan", "null"}:
        return ""
    return text


def normalize_type(value):
    text = normalize_text(value).lower()
    if text == "0":
        return "lane"
    if text == "1":
        return "interaction"
    if text == "intersection":
        return "interaction"
    return text


def iter_lines(gdf, allowed_types=None):
    allowed = None if allowed_types is None else {normalize_type(item) for item in allowed_types}
    for _, row in gdf.iterrows():
        row_type = normalize_type(row.get("type", ""))
        if allowed is not None and row_type not in allowed:
            continue
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type == "LineString":
            yield row, geom
        elif geom.geom_type == "MultiLineString":
            for part in geom.geoms:
                if not part.is_empty:
                    yield row, part


def add_or_update_edge(G, u, v, weight, **attrs):
    if u == v:
        return
    if G.has_edge(u, v):
        if weight < G[u][v].get("weight", float("inf")):
            G[u][v].update(weight=weight, **attrs)
    else:
        G.add_edge(u, v, weight=weight, **attrs)


def classify_event_node(indeg, outdeg):
    total_degree = indeg + outdeg
    if total_degree == 0:
        return "isolated"
    if total_degree == 1:
        return "boundary"
    if indeg == 1 and outdeg == 1:
        return "pass_through"
    if (indeg == 2 and outdeg == 1) or (indeg == 1 and outdeg == 2):
        return "lane_count_change"
    if total_degree >= 3:
        return "intersection"
    return "complex"


def annotate_event_attrs(G):
    for node in G.nodes():
        indeg = G.in_degree(node)
        outdeg = G.out_degree(node)
        event_type = G.nodes[node].get("raw_event_type")
        if event_type is None:
            event_type = classify_event_node(indeg, outdeg)
        G.nodes[node].update(
            event_type=event_type,
            indegree=indeg,
            outdegree=outdeg,
        )
    return G


def parse_center_xy(center_value):
    text = normalize_text(center_value)
    if not text:
        return None
    parts = text.split(",")
    if len(parts) < 2:
        return None
    try:
        return (float(parts[0]), float(parts[1]))
    except ValueError:
        return None


def build_field_graph(gdf):
    """用 source/target 原生字段建图；这是预测图和带节点真值图的首选路径。"""
    if "source" not in gdf.columns or "target" not in gdf.columns:
        return None, set()

    G = nx.DiGraph()
    node_xy_candidates = {}
    protected_nodes = set()

    for row, geom in iter_lines(gdf, allowed_types=None):
        row_type = normalize_type(row.get("type", ""))
        lane_type = normalize_type(row.get("lane_type", ""))
        semantic_type = "interaction" if row_type == "interaction" or lane_type == "interaction" else row_type
        feature_id = get_feature_id(row)
        source = normalize_text(row.get("source", ""))
        target = normalize_text(row.get("target", ""))
        if not source or not target or source == target:
            continue

        coords = list(geom.coords)
        source_xy = (coords[0][0], coords[0][1])
        target_xy = (coords[-1][0], coords[-1][1])
        node_xy_candidates.setdefault(source, []).append(source_xy)
        node_xy_candidates.setdefault(target, []).append(target_xy)

        if semantic_type == "interaction":
            protected_nodes.update([source, target])
        add_or_update_edge(
            G,
            source,
            target,
            float(geom.length),
            edge_type=semantic_type,
            lane_types=(lane_type,),
            feature_ids=(feature_id,),
            raw_edge_count=1,
            geometry=[(coord[0], coord[1]) for coord in geom.coords],
        )

    for node, coords in node_xy_candidates.items():
        xy = np.asarray(coords, dtype=float)
        G.nodes[node]["xy"] = (float(xy[:, 0].mean()), float(xy[:, 1].mean()))

    return G, protected_nodes


def get_feature_id(row):
    feat_id = normalize_text(row.get("feat_id", ""))
    if feat_id:
        return feat_id

    center = normalize_text(row.get("center", ""))
    if center and center != "0":
        return f"center:{center}"

    label = normalize_text(row.get("Label", ""))
    lane_number = normalize_text(row.get("LaneNumber", ""))
    lane_type = normalize_type(row.get("lane_type", ""))
    row_type = normalize_type(row.get("type", ""))
    if label or lane_number:
        return f"group:{row_type}:{lane_type}:{label}:{lane_number}"

    feat_id = normalize_text(row.get("id", ""))
    if feat_id:
        return feat_id
    feat_id = normalize_text(row.get("id_1", ""))
    if feat_id:
        return feat_id
    return normalize_text(row.get("Id", ""))


def cluster_points(points, tolerance):
    if not points:
        return {}
    tree = KDTree(points)
    point_to_node = {}
    visited = set()
    for i, point in enumerate(points):
        if i in visited:
            continue
        indices = tree.query_ball_point(point, tolerance)
        cluster = [points[idx] for idx in indices]
        centroid = (
            float(np.mean([p[0] for p in cluster])),
            float(np.mean([p[1] for p in cluster])),
        )
        for idx in indices:
            visited.add(idx)
            point_to_node[points[idx]] = centroid
    return point_to_node


def has_node_fields(gdf):
    return "source" in gdf.columns and "target" in gdf.columns


def has_valid_node_assignments(gdf):
    if not has_node_fields(gdf):
        return False

    has_line = False
    for row, geom in iter_lines(gdf, allowed_types=None):
        has_line = True
        source = normalize_text(row.get("source", ""))
        target = normalize_text(row.get("target", ""))
        if not (source and target and source != target and source != "0" and target != "0"):
            return False
    return has_line


def annotate_lines_with_nodes(
    input_shp,
    output_shp,
    target_crs=TARGET_CRS,
    merge_tolerance=GT_NODE_MERGE_TOLERANCE,
    intersection_buffer=GT_INTERSECTION_BUFFER,
    feature_prefix="graph",
):
    """
    为线要素补 source/target/lane_type 字段。
    source/target 使用原始折线相邻顶点方向，不反转线段。
    原始折线转折点被视为真实节点，因此一条多顶点折线会拆成多条有向边。
    若原文件已存在有效 source/target，则直接返回原数据。
    """
    source_gdf = read_metric_gdf(input_shp, target_crs=target_crs)
    if has_valid_node_assignments(source_gdf):
        return source_gdf, input_shp

    node_points = []
    segment_records = []
    for feature_idx, (_, row) in enumerate(source_gdf.iterrows(), start=1):
        geom = row.geometry
        if geom is None or geom.is_empty or geom.geom_type != "LineString":
            continue
        coords = list(geom.coords)
        if len(coords) < 2:
            continue
        row_type = normalize_type(row.get("type", ""))
        line_points = [(coord[0], coord[1]) for coord in coords]
        node_points.extend(line_points)
        for seg_idx, (start_coord, end_coord) in enumerate(zip(line_points[:-1], line_points[1:])):
            start = (start_coord[0], start_coord[1])
            end = (end_coord[0], end_coord[1])
            if start == end:
                continue
            is_first_segment = seg_idx == 0
            is_last_segment = seg_idx == len(line_points) - 2
            segment_records.append(
                (
                    row,
                    LineString([start, end]),
                    start,
                    end,
                    row_type,
                    feature_idx,
                    is_first_segment,
                    is_last_segment,
                )
            )

    point_to_node = cluster_points(node_points, merge_tolerance)
    unique_nodes = sorted(set(point_to_node.values()), key=lambda pt: (pt[0], pt[1]))
    node_ids = {node: f"node_{idx + 1}" for idx, node in enumerate(unique_nodes)}

    annotated_rows = []
    new_id = 1
    for row, geom, start, end, row_type, feature_idx, is_first_segment, is_last_segment in segment_records:
        source_node_xy = point_to_node[start]
        target_node_xy = point_to_node[end]
        source_node = node_ids[source_node_xy]
        target_node = node_ids[target_node_xy]

        near_intersection = row_type == "interaction"
        original_lane_type = normalize_text(row.get("lane_type", ""))
        inherited_lane_type = original_lane_type if original_lane_type and original_lane_type != "0" else ""

        new_row = row.copy()
        new_row["type"] = "interaction" if near_intersection else "lane"
        new_row["lane_type"] = "intersection" if near_intersection else (inherited_lane_type or "lane")
        new_row["source"] = source_node
        new_row["target"] = target_node
        new_row["src_anchor"] = 1 if is_first_segment else 0
        new_row["tgt_anchor"] = 1 if is_last_segment else 0
        new_row["feat_id"] = f"{feature_prefix}_{feature_idx}"
        new_row["id"] = new_id
        new_row["geometry"] = geom
        annotated_rows.append(new_row)
        new_id += 1

    annotated_gdf = gpd.GeoDataFrame(annotated_rows, geometry="geometry", crs=source_gdf.crs)
    output_path = Path(output_shp)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    annotated_gdf.to_file(output_path)
    return annotated_gdf, str(output_path)


def build_geometry_graph(gdf, merge_tolerance=0.1):
    """
    GT 没有 source/target 时的兜底路径。
    使用所有折线坐标建图，并把靠近 type=1(interaction) 线的节点作为局部拓扑保护节点。
    """
    raw_points = []
    edge_records = []
    interaction_geoms = []

    for row, geom in iter_lines(gdf, allowed_types=None):
        row_type = normalize_type(row.get("type", ""))
        coords = [(coord[0], coord[1]) for coord in geom.coords]
        if len(coords) < 2:
            continue
        if row_type in {"1", "interaction"}:
            interaction_geoms.append(geom)
        raw_points.extend(coords)
        for start, end in zip(coords[:-1], coords[1:]):
            edge_records.append(
                (
                    start,
                    end,
                    float(LineString([start, end]).length),
                    "interaction" if row_type in {"1", "interaction"} else "lane",
                    "geometry_lane",
                )
            )

    point_to_node = cluster_points(raw_points, merge_tolerance)
    G = nx.DiGraph()
    for start_raw, end_raw, weight, edge_type, lane_type in edge_records:
        start = point_to_node[start_raw]
        end = point_to_node[end_raw]
        add_or_update_edge(
            G,
            start,
            end,
            weight,
            edge_type=edge_type,
            lane_types=(lane_type,),
            feature_ids=("geometry",),
            raw_edge_count=1,
            geometry=[start, end],
        )
        G.nodes[start]["xy"] = start
        G.nodes[end]["xy"] = end

    protected_nodes = set()
    if interaction_geoms:
        for node in G.nodes():
            point = Point(G.nodes[node]["xy"])
            if any(point.distance(interaction) <= 435.0 for interaction in interaction_geoms):
                protected_nodes.add(node)

    return G, protected_nodes


def edge_is_invalid_lane_chain(attrs):
    return False


def is_compressible_pass_through(G, node):
    """
    只压缩由锚点切分出的线段内部中间点。
    是否保留节点由 find_compression_anchors 统一决定；这里仅判断它是否是普通 1-in-1-out 点。
    """
    return G.in_degree(node) == 1 and G.out_degree(node) == 1


def find_junction_centers(G):
    """路口中心车道线交汇处：入大于 2 或者出大于 2。"""
    return {
        node
        for node in G.nodes()
        if G.in_degree(node) >= 3 or G.out_degree(node) >= 3
    }


def find_lane_intersection_nodes(G, junction_centers):
    """
    路口之间的车道线交汇点：
    保留车道数量变化导致的 merge/split 点，以及其它非普通 1-in-1-out 的交汇点，
    但排除已经单独作为“路口中心”处理的节点。
    """
    lane_intersections = set()
    for node in G.nodes():
        if node in junction_centers:
            continue
        indeg = G.in_degree(node)
        outdeg = G.out_degree(node)
        if indeg == 1 and outdeg == 1:
            continue
        if indeg + outdeg <= 1:
            continue
        lane_intersections.add(node)
    return lane_intersections


def find_boundary_nodes(G):
    """保留整条有向链的自然起终点，避免把首尾也压掉。"""
    return {
        node
        for node in G.nodes()
        if G.in_degree(node) == 0 or G.out_degree(node) == 0
    }


def find_compression_anchors(G):
    """
    压缩锚点：
    1. 路口中心车道线交汇处（入大于 2 或出大于 2）；
    2. 与路口中心直接连接的线段起点/终点；
    3. 路口之间因车道数量变化产生的车道线交汇点；
    4. 整条链的边界起终点。
    两个锚点之间构成一条有向线段，中间普通点全部压缩。
    """
    junction_centers = find_junction_centers(G)
    junction_endpoints = set()
    for center in junction_centers:
        # 保留与路口中心直接连接的车道端点；方向仍由原始有向边决定。
        junction_endpoints.update(G.predecessors(center))
        junction_endpoints.update(G.successors(center))
    lane_intersection_nodes = find_lane_intersection_nodes(G, junction_centers)
    boundary_nodes = find_boundary_nodes(G)

    anchors = set()
    anchors.update(junction_centers)
    anchors.update(junction_endpoints)
    anchors.update(lane_intersection_nodes)
    anchors.update(boundary_nodes)
    return anchors, junction_centers, junction_endpoints, lane_intersection_nodes, boundary_nodes


def compress_one_pass(G, protected_nodes):
    anchors, junction_centers, junction_endpoints, lane_intersection_nodes, boundary_nodes = find_compression_anchors(G)
    if not anchors:
        return annotate_event_attrs(G.copy())

    H = nx.DiGraph()
    H.graph.update(G.graph)
    for node in anchors:
        H.add_node(node, **G.nodes[node])
        H.nodes[node]["raw_event_type"] = classify_event_node(G.in_degree(node), G.out_degree(node))
        H.nodes[node]["raw_indegree"] = G.in_degree(node)
        H.nodes[node]["raw_outdegree"] = G.out_degree(node)
        H.nodes[node]["is_junction_center"] = node in junction_centers
        H.nodes[node]["is_junction_endpoint"] = node in junction_endpoints
        H.nodes[node]["is_lane_intersection"] = node in lane_intersection_nodes
        H.nodes[node]["is_boundary_node"] = node in boundary_nodes

    for start in anchors:
        for succ in G.successors(start):
            current = succ
            edge_attrs = G[start][succ]
            total_weight = edge_attrs["weight"]
            edge_types = [edge_attrs.get("edge_type", "")]
            lane_types = list(edge_attrs.get("lane_types", ()))
            feature_ids = list(edge_attrs.get("feature_ids", ()))
            raw_edge_count = edge_attrs.get("raw_edge_count", 1)
            path_geometry = list(edge_attrs.get("geometry", [node_xy(G, start), node_xy(G, succ)]))
            visited = {start}

            while current not in anchors and current not in visited:
                visited.add(current)
                next_nodes = list(G.successors(current))
                if len(next_nodes) != 1:
                    break
                nxt = next_nodes[0]
                attrs = G[current][nxt]
                total_weight += attrs["weight"]
                edge_types.append(attrs.get("edge_type", ""))
                lane_types.extend(attrs.get("lane_types", ()))
                feature_ids.extend(attrs.get("feature_ids", ()))
                raw_edge_count += attrs.get("raw_edge_count", 1)
                next_geometry = list(attrs.get("geometry", [node_xy(G, current), node_xy(G, nxt)]))
                if path_geometry and next_geometry and path_geometry[-1] == next_geometry[0]:
                    path_geometry.extend(next_geometry[1:])
                else:
                    path_geometry.extend(next_geometry)
                current = nxt

            if current not in anchors or current == start:
                continue

            compressed_type = "interaction" if "interaction" in edge_types else "lane"
            attrs = {
                "edge_type": compressed_type,
                "lane_types": tuple(lane_types),
                "feature_ids": tuple(sorted(set(feature_ids))),
                "raw_edge_count": raw_edge_count,
                "geometry": path_geometry,
            }
            if edge_is_invalid_lane_chain(attrs):
                continue
            add_or_update_edge(H, start, current, total_weight, **attrs)

    return annotate_event_attrs(H)


def compress_native_graph(G, protected_nodes):
    """
    只压缩原始图中本来就是 1-in-1-out 的过路点。
    原始图中的端点、merge/split、路口节点即使压缩后变成 1-in-1-out，也保留其原始拓扑身份。
    """
    return annotate_event_attrs(compress_one_pass(G, protected_nodes))


def build_native_topology(gdf, name):
    field_graph, protected_nodes = build_field_graph(gdf)
    if field_graph is not None and field_graph.number_of_edges() > 0:
        print(f"    - {name}: 使用 source/target 原生拓扑字段建图")
        compressed = compress_native_graph(field_graph, protected_nodes)
        return compressed, field_graph, protected_nodes

    print(f"    - {name}: 未发现 source/target 字段，使用几何兜底建图")
    geometry_graph, protected_nodes = build_geometry_graph(gdf)
    compressed = compress_native_graph(geometry_graph, protected_nodes)
    return compressed, geometry_graph, protected_nodes


def node_xy(G, node):
    return G.nodes[node].get("xy", node)


def edge_heading_deg(G, u, v):
    p0 = np.asarray(node_xy(G, u), dtype=float)
    p1 = np.asarray(node_xy(G, v), dtype=float)
    vec = p1 - p0
    norm = np.linalg.norm(vec)
    if norm < 1e-9:
        return None
    return float(np.degrees(np.arctan2(vec[1], vec[0])) % 360.0)


def circular_angle_diff_deg(a, b):
    diff = abs(a - b) % 360.0
    return min(diff, 360.0 - diff)


def ordered_incident_headings(G, node):
    incoming = []
    outgoing = []
    for src in G.predecessors(node):
        heading = edge_heading_deg(G, src, node)
        if heading is not None:
            incoming.append(heading)
    for dst in G.successors(node):
        heading = edge_heading_deg(G, node, dst)
        if heading is not None:
            outgoing.append(heading)
    return sorted(incoming), sorted(outgoing)


def ordered_heading_penalty(source_angles, target_angles):
    if not source_angles and not target_angles:
        return 0.0
    if not source_angles or not target_angles:
        return 1.0

    def sequence_score(a_seq, b_seq):
        count = min(len(a_seq), len(b_seq))
        scores = []
        for i in range(count):
            diff = circular_angle_diff_deg(a_seq[i], b_seq[i])
            # Small heading deviations are treated as equivalent topology.
            if diff <= ANGLE_TOLERANCE_DEG:
                scores.append(0.0)
            else:
                scores.append((diff - ANGLE_TOLERANCE_DEG) / max(1.0, 180.0 - ANGLE_TOLERANCE_DEG))
        scores.extend([1.0] * abs(len(a_seq) - len(b_seq)))
        return float(np.mean(scores)) if scores else 0.0

    best = float("inf")
    longer = target_angles if len(target_angles) >= len(source_angles) else source_angles
    for shift in range(len(longer)):
        if len(target_angles) >= len(source_angles):
            rotated = target_angles[shift:] + target_angles[:shift]
            best = min(best, sequence_score(source_angles, rotated))
        else:
            rotated = source_angles[shift:] + source_angles[:shift]
            best = min(best, sequence_score(rotated, target_angles))
    return best


def edge_endpoint_signature(G, neighbor, is_incoming):
    attrs = G.nodes[neighbor]
    return {
        "event_type": attrs.get("event_type"),
        "indegree": G.in_degree(neighbor),
        "outdegree": G.out_degree(neighbor),
        "is_incoming": is_incoming,
    }


def incident_edge_records(G, node):
    incoming = []
    outgoing = []
    for src in G.predecessors(node):
        incoming.append(
            {
                "heading": edge_heading_deg(G, src, node),
                "neighbor": src,
                "neighbor_sig": edge_endpoint_signature(G, src, True),
            }
        )
    for dst in G.successors(node):
        outgoing.append(
            {
                "heading": edge_heading_deg(G, node, dst),
                "neighbor": dst,
                "neighbor_sig": edge_endpoint_signature(G, dst, False),
            }
        )
    return incoming, outgoing


def single_incident_edge_cost(src_edge, tgt_edge):
    src_heading = src_edge.get("heading")
    tgt_heading = tgt_edge.get("heading")
    if src_heading is None or tgt_heading is None:
        heading_cost = 1.0
    else:
        diff = circular_angle_diff_deg(src_heading, tgt_heading)
        if diff <= ANGLE_TOLERANCE_DEG:
            heading_cost = 0.0
        else:
            heading_cost = (diff - ANGLE_TOLERANCE_DEG) / max(1.0, 180.0 - ANGLE_TOLERANCE_DEG)

    src_sig = src_edge["neighbor_sig"]
    tgt_sig = tgt_edge["neighbor_sig"]
    type_cost = 0.0 if src_sig["event_type"] == tgt_sig["event_type"] else 1.0
    degree_cost = abs(src_sig["indegree"] - tgt_sig["indegree"])
    degree_cost += abs(src_sig["outdegree"] - tgt_sig["outdegree"])
    return heading_cost + type_cost + 0.35 * degree_cost


def incident_edge_set_cost(src_edges, tgt_edges):
    if not src_edges and not tgt_edges:
        return 0.0
    if not src_edges or not tgt_edges:
        return float(abs(len(src_edges) - len(tgt_edges)) + min(len(src_edges), len(tgt_edges)))

    size = max(len(src_edges), len(tgt_edges))
    dummy_cost = 3.0
    cost_matrix = np.full((size, size), dummy_cost, dtype=float)
    for i, src_edge in enumerate(src_edges):
        for j, tgt_edge in enumerate(tgt_edges):
            cost_matrix[i, j] = single_incident_edge_cost(src_edge, tgt_edge)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    total_cost = 0.0
    for row, col in zip(row_ind, col_ind):
        total_cost += cost_matrix[row, col]
    return total_cost / max(len(src_edges), len(tgt_edges))


def node_and_incident_edge_cost(G_source, src, G_target, tgt, search_radius, dist_weight=1.0, topo_weight=2.0):
    src_xy = node_xy(G_source, src)
    tgt_xy = node_xy(G_target, tgt)
    dist = float(np.linalg.norm(np.asarray(src_xy) - np.asarray(tgt_xy)))

    degree_penalty = abs(G_source.in_degree(src) - G_target.in_degree(tgt))
    degree_penalty += abs(G_source.out_degree(src) - G_target.out_degree(tgt))
    type_penalty = (
        0.0
        if G_source.nodes[src].get("event_type") == G_target.nodes[tgt].get("event_type")
        else 5.0
    )

    src_in, src_out = incident_edge_records(G_source, src)
    tgt_in, tgt_out = incident_edge_records(G_target, tgt)
    incoming_cost = incident_edge_set_cost(src_in, tgt_in)
    outgoing_cost = incident_edge_set_cost(src_out, tgt_out)
    structure_cost = degree_penalty + type_penalty + incoming_cost + outgoing_cost
    return 0.25 * dist_weight * (dist / max(search_radius, 1e-6)) + topo_weight * structure_cost


def topology_aware_node_match(G_source, G_target, tau=1.5, dist_weight=1.0, topo_weight=2.0):
    source_nodes = list(G_source.nodes())
    target_nodes = list(G_target.nodes())
    if not source_nodes or not target_nodes:
        return {}, {}

    target_points = [node_xy(G_target, node) for node in target_nodes]
    target_tree = KDTree(target_points)
    search_radius = max(tau, MATCH_SEARCH_RADIUS)
    candidate_map = {}

    for src in source_nodes:
        src_xy = node_xy(G_source, src)
        indices = target_tree.query_ball_point(src_xy, search_radius)
        if not indices:
            k = min(MATCH_NEAREST_CANDIDATES, len(target_nodes))
            _, nearest = target_tree.query(src_xy, k=k)
            if np.isscalar(nearest):
                indices = [int(nearest)]
            else:
                indices = [int(idx) for idx in np.atleast_1d(nearest)]
        candidate_map[src] = sorted(set(target_nodes[idx] for idx in indices))

    source_index = {node: idx for idx, node in enumerate(source_nodes)}
    target_index = {node: idx for idx, node in enumerate(target_nodes)}
    size = max(len(source_nodes), len(target_nodes))
    dummy_cost = 25.0
    cost_matrix = np.full((size, size), dummy_cost, dtype=float)

    for src in source_nodes:
        row = source_index[src]
        for tgt in candidate_map.get(src, []):
            col = target_index[tgt]
            cost_matrix[row, col] = node_and_incident_edge_cost(
                G_source,
                src,
                G_target,
                tgt,
                search_radius,
                dist_weight=dist_weight,
                topo_weight=topo_weight,
            )

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    source_to_target = {}
    target_to_source = {}
    for row, col in zip(row_ind, col_ind):
        if row >= len(source_nodes) or col >= len(target_nodes):
            continue
        if cost_matrix[row, col] >= dummy_cost:
            continue
        src = source_nodes[row]
        tgt = target_nodes[col]
        source_to_target[src] = tgt
        target_to_source[tgt] = src
    refine_matches_by_neighbors(
        G_source,
        G_target,
        source_to_target,
        target_to_source,
        tau=tau,
        dist_weight=dist_weight,
        topo_weight=topo_weight,
    )
    return source_to_target, target_to_source


def refine_matches_by_neighbors(
    G_source,
    G_target,
    source_to_target,
    target_to_source,
    tau=1.5,
    dist_weight=1.0,
    topo_weight=2.0,
    max_rounds=6,
):
    """
    二次补匹配：
    对于首轮没匹配上的节点，如果它和若干已匹配邻居的连接关系在另一张图中能唯一约束出候选点，
    则把这种“拓扑上显然对应”的点补回来。
    """

    def candidate_targets_from_matched_neighbors(src):
        candidate_sets = []

        mapped_in_neighbors = [
            source_to_target[pred]
            for pred in G_source.predecessors(src)
            if pred in source_to_target
        ]
        mapped_out_neighbors = [
            source_to_target[succ]
            for succ in G_source.successors(src)
            if succ in source_to_target
        ]

        for tgt_pred in mapped_in_neighbors:
            candidate_sets.append(set(G_target.successors(tgt_pred)))
        for tgt_succ in mapped_out_neighbors:
            candidate_sets.append(set(G_target.predecessors(tgt_succ)))

        if not candidate_sets:
            return set()

        candidates = set.intersection(*candidate_sets) if candidate_sets else set()
        candidates = {node for node in candidates if node not in target_to_source}
        return candidates

    for _ in range(max_rounds):
        progress = False
        for src in G_source.nodes():
            if src in source_to_target:
                continue

            candidates = candidate_targets_from_matched_neighbors(src)
            if not candidates:
                continue

            scored = []
            src_xy = node_xy(G_source, src)
            search_radius = max(tau, MATCH_SEARCH_RADIUS)
            for tgt in candidates:
                tgt_xy = node_xy(G_target, tgt)
                dist = float(np.linalg.norm(np.asarray(src_xy) - np.asarray(tgt_xy)))
                src_in, src_out = ordered_incident_headings(G_source, src)
                tgt_in, tgt_out = ordered_incident_headings(G_target, tgt)
                degree_penalty = abs(G_source.in_degree(src) - G_target.in_degree(tgt))
                degree_penalty += abs(G_source.out_degree(src) - G_target.out_degree(tgt))
                type_penalty = (
                    0.0
                    if G_source.nodes[src].get("event_type") == G_target.nodes[tgt].get("event_type")
                    else 5.0
                )
                angle_penalty = ordered_heading_penalty(src_in, tgt_in)
                angle_penalty += ordered_heading_penalty(src_out, tgt_out)
                total_cost = 0.35 * dist_weight * (dist / max(search_radius, 1e-6)) + topo_weight * (
                    degree_penalty + type_penalty + angle_penalty
                )
                scored.append((total_cost, tgt))

            if not scored:
                continue

            scored.sort(key=lambda item: item[0])
            best_cost, best_tgt = scored[0]
            second_cost = scored[1][0] if len(scored) > 1 else None

            # 拓扑传播补匹配要比首轮更保守：要求候选本身足够好，且和第二名有区分度。
            if best_cost > 4.0:
                continue
            if second_cost is not None and second_cost - best_cost < 0.75:
                continue

            source_to_target[src] = best_tgt
            target_to_source[best_tgt] = src
            progress = True

        if not progress:
            break


def calc_global_apls(G_source, G_target, source_to_target):
    matched_items = list(source_to_target.items())
    if len(matched_items) < 2:
        return 0.0
    source_paths = dict(nx.all_pairs_dijkstra_path_length(G_source, weight="weight"))
    target_paths = dict(nx.all_pairs_dijkstra_path_length(G_target, weight="weight"))
    scores = []
    for u_src, u_tgt in matched_items:
        for v_src, v_tgt in matched_items:
            if u_src == v_src:
                continue
            source_length = source_paths.get(u_src, {}).get(v_src)
            if source_length is None or source_length <= 0:
                continue
            target_length = target_paths.get(u_tgt, {}).get(v_tgt)
            if target_length is None:
                scores.append(0.0)
            else:
                scores.append(1.0 - min(1.0, abs(target_length - source_length) / source_length))
    return float(np.mean(scores)) if scores else 0.0


def evaluate_directed_topology(G_gt, G_pred, tau=1.5):
    match_pred_to_gt, match_gt_to_pred = topology_aware_node_match(G_pred, G_gt, tau=tau)
    gt_edges = set(G_gt.edges())
    pred_edges = set(G_pred.edges())

    matched_pred_edges = {
        (u, v)
        for u, v in pred_edges
        if match_pred_to_gt.get(u) is not None
        and match_pred_to_gt.get(v) is not None
        and (match_pred_to_gt[u], match_pred_to_gt[v]) in gt_edges
    }
    matched_gt_edges = {
        (u, v)
        for u, v in gt_edges
        if match_gt_to_pred.get(u) is not None
        and match_gt_to_pred.get(v) is not None
        and (match_gt_to_pred[u], match_gt_to_pred[v]) in pred_edges
    }

    precision = len(matched_pred_edges) / len(pred_edges) if pred_edges else 0.0
    recall = len(matched_gt_edges) / len(gt_edges) if gt_edges else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    raw_apls = 0.5 * (
        calc_global_apls(G_gt, G_pred, match_gt_to_pred)
        + calc_global_apls(G_pred, G_gt, match_pred_to_gt)
    )
    coverage = min(
        len(match_pred_to_gt) / G_pred.number_of_nodes() if G_pred.number_of_nodes() else 0.0,
        len(match_gt_to_pred) / G_gt.number_of_nodes() if G_gt.number_of_nodes() else 0.0,
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "apls": raw_apls * coverage,
        "raw_apls": raw_apls,
        "coverage": coverage,
        "c_node": coverage,
        "matched_pred_edges": matched_pred_edges,
        "matched_gt_edges": matched_gt_edges,
        "match_pred_to_gt": match_pred_to_gt,
        "match_gt_to_pred": match_gt_to_pred,
    }


def collect_endpoint_matched_but_edge_unmatched(G_gt, G_pred, metrics):
    match_pred_to_gt = metrics.get("match_pred_to_gt", {})
    match_gt_to_pred = metrics.get("match_gt_to_pred", {})
    matched_pred_edges = metrics.get("matched_pred_edges", set())
    matched_gt_edges = metrics.get("matched_gt_edges", set())
    gt_edges = set(G_gt.edges())
    pred_edges = set(G_pred.edges())

    pred_side = []
    for u, v in pred_edges:
        if (u, v) in matched_pred_edges:
            continue
        if u not in match_pred_to_gt or v not in match_pred_to_gt:
            continue
        gt_u = match_pred_to_gt[u]
        gt_v = match_pred_to_gt[v]
        pred_side.append(
            {
                "pred_u": u,
                "pred_v": v,
                "gt_u": gt_u,
                "gt_v": gt_v,
                "gt_edge_exists": (gt_u, gt_v) in gt_edges,
                "gt_reverse_exists": (gt_v, gt_u) in gt_edges,
            }
        )

    gt_side = []
    for u, v in gt_edges:
        if (u, v) in matched_gt_edges:
            continue
        if u not in match_gt_to_pred or v not in match_gt_to_pred:
            continue
        pred_u = match_gt_to_pred[u]
        pred_v = match_gt_to_pred[v]
        gt_side.append(
            {
                "gt_u": u,
                "gt_v": v,
                "pred_u": pred_u,
                "pred_v": pred_v,
                "pred_edge_exists": (pred_u, pred_v) in pred_edges,
                "pred_reverse_exists": (pred_v, pred_u) in pred_edges,
            }
        )

    return pred_side, gt_side


def write_edge_debug_report(name, G_gt, G_pred, metrics, output_path):
    pred_side, gt_side = collect_endpoint_matched_but_edge_unmatched(G_gt, G_pred, metrics)
    lines = [
        f"{name} edge debug report",
        "",
        f"matched_pred_edges={len(metrics.get('matched_pred_edges', []))}",
        f"matched_gt_edges={len(metrics.get('matched_gt_edges', []))}",
        f"pred_endpoint_matched_but_edge_unmatched={len(pred_side)}",
        f"gt_endpoint_matched_but_edge_unmatched={len(gt_side)}",
        "",
        "[Pred side]",
    ]

    for item in pred_side:
        lines.append(
            "pred "
            f"{item['pred_u']} -> {item['pred_v']}  |  "
            f"mapped_gt {item['gt_u']} -> {item['gt_v']}  |  "
            f"gt_edge_exists={item['gt_edge_exists']}  "
            f"gt_reverse_exists={item['gt_reverse_exists']}"
        )

    lines.extend(["", "[GT side]"])
    for item in gt_side:
        lines.append(
            "gt "
            f"{item['gt_u']} -> {item['gt_v']}  |  "
            f"mapped_pred {item['pred_u']} -> {item['pred_v']}  |  "
            f"pred_edge_exists={item['pred_edge_exists']}  "
            f"pred_reverse_exists={item['pred_reverse_exists']}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def collect_unmatched_nodes_with_matched_neighbors(G_gt, G_pred, metrics):
    match_pred_to_gt = metrics.get("match_pred_to_gt", {})
    match_gt_to_pred = metrics.get("match_gt_to_pred", {})

    pred_side = []
    for pred_node in G_pred.nodes():
        if pred_node in match_pred_to_gt:
            continue
        mapped_in = [match_pred_to_gt[n] for n in G_pred.predecessors(pred_node) if n in match_pred_to_gt]
        mapped_out = [match_pred_to_gt[n] for n in G_pred.successors(pred_node) if n in match_pred_to_gt]
        if not mapped_in and not mapped_out:
            continue
        candidate_sets = []
        for tgt_pred in mapped_in:
            candidate_sets.append(set(G_gt.successors(tgt_pred)))
        for tgt_succ in mapped_out:
            candidate_sets.append(set(G_gt.predecessors(tgt_succ)))
        implied = set.intersection(*candidate_sets) if candidate_sets else set()
        implied = [node for node in implied if node not in match_gt_to_pred]
        pred_side.append(
            {
                "pred_node": pred_node,
                "mapped_in_neighbors": mapped_in,
                "mapped_out_neighbors": mapped_out,
                "implied_gt_candidates": implied,
            }
        )

    gt_side = []
    for gt_node in G_gt.nodes():
        if gt_node in match_gt_to_pred:
            continue
        mapped_in = [match_gt_to_pred[n] for n in G_gt.predecessors(gt_node) if n in match_gt_to_pred]
        mapped_out = [match_gt_to_pred[n] for n in G_gt.successors(gt_node) if n in match_gt_to_pred]
        if not mapped_in and not mapped_out:
            continue
        candidate_sets = []
        for pred_pred in mapped_in:
            candidate_sets.append(set(G_pred.successors(pred_pred)))
        for pred_succ in mapped_out:
            candidate_sets.append(set(G_pred.predecessors(pred_succ)))
        implied = set.intersection(*candidate_sets) if candidate_sets else set()
        implied = [node for node in implied if node not in match_pred_to_gt]
        gt_side.append(
            {
                "gt_node": gt_node,
                "mapped_in_neighbors": mapped_in,
                "mapped_out_neighbors": mapped_out,
                "implied_pred_candidates": implied,
            }
        )
    return pred_side, gt_side


def write_unmatched_node_debug_report(name, G_gt, G_pred, metrics, output_path):
    pred_side, gt_side = collect_unmatched_nodes_with_matched_neighbors(G_gt, G_pred, metrics)
    lines = [
        f"{name} unmatched node debug report",
        "",
        f"pred_unmatched_with_matched_neighbors={len(pred_side)}",
        f"gt_unmatched_with_matched_neighbors={len(gt_side)}",
        "",
        "[Pred side]",
    ]
    for item in pred_side:
        lines.append(
            f"pred_node {item['pred_node']} | "
            f"mapped_in={item['mapped_in_neighbors']} | "
            f"mapped_out={item['mapped_out_neighbors']} | "
            f"implied_gt_candidates={item['implied_gt_candidates']}"
        )
    lines.extend(["", "[GT side]"])
    for item in gt_side:
        lines.append(
            f"gt_node {item['gt_node']} | "
            f"mapped_in={item['mapped_in_neighbors']} | "
            f"mapped_out={item['mapped_out_neighbors']} | "
            f"implied_pred_candidates={item['implied_pred_candidates']}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_unmatched_nodes_with_matched_neighbors(
    G_gt,
    G_pred,
    metrics,
    title,
    save_path,
    gt_color="#0072B2",
    pred_color="#D55E00",
    helper_color="#7A7A7A",
    major_step=None,
    bounds_override=None,
):
    pred_side, gt_side = collect_unmatched_nodes_with_matched_neighbors(G_gt, G_pred, metrics)
    match_pred_to_gt = metrics.get("match_pred_to_gt", {})
    match_gt_to_pred = metrics.get("match_gt_to_pred", {})

    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)

    gt_nodes = [item["gt_node"] for item in gt_side]
    if gt_nodes:
        pts = np.asarray([node_xy(G_gt, node) for node in gt_nodes], dtype=float)
        ax.scatter(pts[:, 0], pts[:, 1], s=16, c=gt_color, marker="s", linewidths=0, zorder=4)

    pred_nodes = [item["pred_node"] for item in pred_side]
    if pred_nodes:
        pts = np.asarray([node_xy(G_pred, node) for node in pred_nodes], dtype=float)
        ax.scatter(pts[:, 0], pts[:, 1], s=16, c=pred_color, marker="x", linewidths=0.85, zorder=5)

    for item in pred_side:
        px, py = node_xy(G_pred, item["pred_node"])
        for tgt in item["implied_gt_candidates"]:
            gx, gy = node_xy(G_gt, tgt)
            ax.plot([px, gx], [py, gy], linestyle="--", color=helper_color, linewidth=0.45, alpha=0.25, zorder=2)

    for item in gt_side:
        gx, gy = node_xy(G_gt, item["gt_node"])
        for pred in item["implied_pred_candidates"]:
            px, py = node_xy(G_pred, pred)
            ax.plot([gx, px], [gy, py], linestyle="--", color=helper_color, linewidth=0.45, alpha=0.25, zorder=2)

    for pred_node, gt_node in match_pred_to_gt.items():
        if pred_node in pred_nodes:
            continue
        px, py = node_xy(G_pred, pred_node)
        gx, gy = node_xy(G_gt, gt_node)
        ax.scatter([px], [py], s=7, c="#F2BE92", linewidths=0, alpha=0.28, zorder=1)
        ax.scatter([gx], [gy], s=7, c="#A9C8E6", linewidths=0, alpha=0.28, zorder=1)

    legend_handles = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor=gt_color, markersize=7, label="GT unmatched node"),
        Line2D([0], [0], marker="x", color=pred_color, markersize=7, label="Ours unmatched node"),
        Line2D([0], [0], color=helper_color, lw=1, ls="--", label="Implied candidate link"),
    ]
    legend = ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        frameon=True,
        fontsize=LEGEND_FONTSIZE,
    )
    bounds = bounds_override if bounds_override is not None else graph_bounds(G_gt, G_pred)
    apply_grid_paper_axes(ax, title, bounds, legend, major_step=major_step)
    save_publication_figure(fig, save_path)


def plot_endpoint_matched_but_edge_unmatched(
    G_gt,
    G_pred,
    metrics,
    title,
    save_path,
    gt_color="#0072B2",
    pred_color="#D55E00",
    match_line_color="#7A7A7A",
    major_step=None,
    bounds_override=None,
):
    pred_side, gt_side = collect_endpoint_matched_but_edge_unmatched(G_gt, G_pred, metrics)
    pred_problem_edges = {(item["pred_u"], item["pred_v"]) for item in pred_side}
    gt_problem_edges = {(item["gt_u"], item["gt_v"]) for item in gt_side}

    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)

    gt_nodes = set()
    for u, v in gt_problem_edges:
        attrs = G_gt[u][v]
        coords = edge_coords(G_gt, u, v, attrs)
        xs = [coord[0] for coord in coords]
        ys = [coord[1] for coord in coords]
        ax.plot(xs, ys, color=gt_color, linewidth=0.95, alpha=0.92, zorder=2, solid_capstyle="round")
        gt_nodes.update([u, v])

    pred_nodes = set()
    for u, v in pred_problem_edges:
        attrs = G_pred[u][v]
        coords = edge_coords(G_pred, u, v, attrs)
        xs = [coord[0] for coord in coords]
        ys = [coord[1] for coord in coords]
        ax.plot(xs, ys, color=pred_color, linewidth=0.95, alpha=0.92, zorder=3, solid_capstyle="round")
        pred_nodes.update([u, v])

    if gt_nodes:
        gt_pts = np.asarray([node_xy(G_gt, node) for node in gt_nodes], dtype=float)
        ax.scatter(gt_pts[:, 0], gt_pts[:, 1], s=15, c=gt_color, marker="s", linewidths=0, zorder=4)
    if pred_nodes:
        pred_pts = np.asarray([node_xy(G_pred, node) for node in pred_nodes], dtype=float)
        ax.scatter(pred_pts[:, 0], pred_pts[:, 1], s=16, c=pred_color, marker="x", linewidths=0.85, zorder=5)

    match_pred_to_gt = metrics.get("match_pred_to_gt", {})
    for item in pred_side:
        pred_u = item["pred_u"]
        pred_v = item["pred_v"]
        gt_u = match_pred_to_gt.get(pred_u)
        gt_v = match_pred_to_gt.get(pred_v)
        if gt_u is not None:
            px, py = node_xy(G_pred, pred_u)
            gx, gy = node_xy(G_gt, gt_u)
            ax.plot([px, gx], [py, gy], linestyle="--", color=match_line_color, linewidth=0.45, alpha=0.22, zorder=1)
        if gt_v is not None:
            px, py = node_xy(G_pred, pred_v)
            gx, gy = node_xy(G_gt, gt_v)
            ax.plot([px, gx], [py, gy], linestyle="--", color=match_line_color, linewidth=0.45, alpha=0.22, zorder=1)

    legend_handles = [
        Line2D([0], [0], color=gt_color, lw=2, label="GT problem edge"),
        Line2D([0], [0], color=pred_color, lw=2, label="Ours problem edge"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=gt_color, markersize=6, label="GT problem node"),
        Line2D([0], [0], marker="x", color=pred_color, markersize=7, label="Ours problem node"),
        Line2D([0], [0], color=match_line_color, lw=1, ls="--", label="Endpoint match link"),
    ]

    legend = ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        frameon=True,
        fontsize=LEGEND_FONTSIZE,
    )
    bounds = bounds_override if bounds_override is not None else graph_bounds(G_gt, G_pred)
    apply_grid_paper_axes(ax, title, bounds, legend, major_step=major_step)
    save_publication_figure(fig, save_path)


def one_hop_region_subgraph(G, seed_nodes):
    """
    仅保留“种子节点 + 与种子直接相连的一跳节点”组成的局部拓扑区域。
    邻居节点不再继续外扩，也不保留邻居之间的额外连接。
    因此一个 3 进 3 出的路口会形成 7 个节点、6 条线的区域。
    """
    H = nx.DiGraph()
    H.graph.update(G.graph)

    for node in seed_nodes:
        if node not in G:
            continue
        H.add_node(node, **G.nodes[node])

        for pred in G.predecessors(node):
            if pred not in H:
                H.add_node(pred, **G.nodes[pred])
            H.add_edge(pred, node, **G[pred][node])

        for succ in G.successors(node):
            if succ not in H:
                H.add_node(succ, **G.nodes[succ])
            H.add_edge(node, succ, **G[node][succ])

    return H


def junction_subgraph(G):
    """
    路口区域：
    只以虚拟路口中心节点作为种子，局部区域 = 中心节点 + 与其直接相连的节点。
    """
    junction_centers = {
        node for node, attrs in G.nodes(data=True) if attrs.get("is_junction_center")
    }
    if not junction_centers:
        junction_centers = {
            node
            for node, attrs in G.nodes(data=True)
            if attrs.get("event_type") == "intersection"
        }
    return one_hop_region_subgraph(G, junction_centers)


def lane_change_subgraph(G):
    """
    变道区域：
    只以变道交汇点作为种子，局部区域 = 变道交汇点 + 与其直接相连的节点。
    """
    lane_change_nodes = {
        node
        for node, attrs in G.nodes(data=True)
        if attrs.get("raw_event_type") == "lane_count_change"
        or (attrs.get("raw_event_type") is None and attrs.get("event_type") == "lane_count_change")
    }
    return one_hop_region_subgraph(G, lane_change_nodes)


def road_segment_subgraph(G):
    """
    道路段区域：
    从整体骨架中仅剔除“事件核心节点”本身：
    1. 虚拟路口中心节点；
    2. 变道交汇点节点。
    与这些核心节点直接相连的普通道路段节点仍然保留，因为它们本质上仍属于道路段。
    在此基础上，再剔除带有路口语义的 interaction 边，避免道路段结果中混入路口线。
    """
    excluded_nodes = {
        node
        for node, attrs in G.nodes(data=True)
        if attrs.get("is_junction_center")
        or attrs.get("raw_event_type") == "lane_count_change"
        or (attrs.get("raw_event_type") is None and attrs.get("event_type") == "lane_count_change")
    }
    kept_nodes = [node for node in G.nodes() if node not in excluded_nodes]

    H = G.subgraph(kept_nodes).copy()
    interaction_edges = []
    for u, v, attrs in H.edges(data=True):
        lane_types = {normalize_type(value) for value in attrs.get("lane_types", ())}
        if attrs.get("edge_type") == "interaction" or "interaction" in lane_types:
            interaction_edges.append((u, v))
    if interaction_edges:
        H.remove_edges_from(interaction_edges)
    isolated = [node for node in H.nodes() if H.in_degree(node) + H.out_degree(node) == 0]
    if isolated:
        H.remove_nodes_from(isolated)
    H.graph.update(G.graph)
    return H


def edge_coords(G, u, v, attrs):
    coords = attrs.get("geometry")
    if coords and len(coords) >= 2:
        return coords
    ux, uy = node_xy(G, u)
    vx, vy = node_xy(G, v)
    return [(ux, uy), (vx, vy)]


def draw_direction_arrow(ax, coords, color, linewidth, alpha=0.9, zorder=5):
    if len(coords) < 2:
        return

    segment_lengths = []
    total_length = 0.0
    for start, end in zip(coords[:-1], coords[1:]):
        seg_len = float(np.linalg.norm(np.asarray(end, dtype=float) - np.asarray(start, dtype=float)))
        segment_lengths.append(seg_len)
        total_length += seg_len

    if total_length <= 1e-9 or total_length < 10.0 or alpha < 0.6:
        return

    target_dist = 0.62 * total_length
    travelled = 0.0
    for (start, end), seg_len in zip(zip(coords[:-1], coords[1:]), segment_lengths):
        if seg_len <= 1e-9:
            continue
        if travelled + seg_len >= target_dist:
            start_pt = np.asarray(start, dtype=float)
            end_pt = np.asarray(end, dtype=float)
            direction = (end_pt - start_pt) / seg_len
            anchor_dist = min(seg_len * 0.9, max(0.0, target_dist - travelled) + seg_len * 0.18)
            arrow_end = start_pt + direction * anchor_dist
            arrow_length = min(seg_len * 0.45, max(total_length * 0.08, 1.2))
            arrow_start = arrow_end - direction * arrow_length
            ax.annotate(
                "",
                xy=(float(arrow_end[0]), float(arrow_end[1])),
                xytext=(float(arrow_start[0]), float(arrow_start[1])),
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": color,
                    "lw": max(0.45, linewidth * 0.55),
                    "alpha": min(alpha, 0.82),
                    "mutation_scale": 8.0,
                    "shrinkA": 0.0,
                    "shrinkB": 0.0,
                },
                zorder=zorder,
            )
            return
        travelled += seg_len


def graph_bounds(*graphs):
    coords = []
    for G in graphs:
        for node in G.nodes():
            coords.append(node_xy(G, node))
    if not coords:
        return None
    pts = np.asarray(coords, dtype=float)
    min_x, min_y = pts.min(axis=0)
    max_x, max_y = pts.max(axis=0)
    width = max(max_x - min_x, 1.0)
    height = max(max_y - min_y, 1.0)
    pad_x = width * LABEL_PADDING_RATIO
    pad_y = height * LABEL_PADDING_RATIO
    return min_x - pad_x, max_x + pad_x, min_y - pad_y, max_y + pad_y


def compute_reference_grid_step(bounds, target_bins=8):
    if bounds is None:
        return None
    min_x, max_x, min_y, max_y = bounds
    span = max(max_x - min_x, max_y - min_y)
    if span <= 0:
        return 1.0
    raw_step = span / max(target_bins, 1)
    exponent = np.floor(np.log10(raw_step))
    scale = 10.0 ** exponent
    normalized = raw_step / scale
    if normalized <= 1.0:
        factor = 1.0
    elif normalized <= 2.0:
        factor = 2.0
    elif normalized <= 5.0:
        factor = 5.0
    else:
        factor = 10.0
    return factor * scale


def finalize_publication_axes(ax, title, bounds=None, legend=None):
    ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight="semibold", pad=10)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor("white")
    if bounds is not None:
        min_x, max_x, min_y, max_y = bounds
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
    if legend is not None:
        frame = legend.get_frame()
        frame.set_facecolor("white")
        frame.set_edgecolor("#D0D7DE")
        frame.set_linewidth(0.8)


def apply_grid_paper_axes(ax, title, bounds=None, legend=None, major_step=None):
    ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight="semibold", pad=10)
    ax.set_aspect("equal")
    ax.set_facecolor("white")
    ax.grid(True, linestyle=":", color="#A0A0A0", alpha=0.6, zorder=0)

    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color("black")

    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        length=5,
        width=1.0,
        labelsize=9,
        bottom=True,
        top=True,
        left=True,
        right=True,
    )
    if major_step is not None and major_step > 0:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(major_step))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(major_step))
    else:
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=8))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=8))
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    if bounds is not None:
        min_x, max_x, min_y, max_y = bounds
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)

    if legend is not None:
        frame = legend.get_frame()
        frame.set_facecolor("white")
        frame.set_edgecolor("#D0D7DE")
        frame.set_linewidth(0.8)


def save_publication_figure(fig, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight", facecolor="white", edgecolor="none")
    fig.savefig(save_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white", edgecolor="none")
    fig.savefig(save_path.with_suffix(".svg"), bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)


def plot_graph(G, title, save_path, edge_color="#245B7A", node_color="#C23B22"):
    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)
    for u, v, attrs in G.edges(data=True):
        coords = edge_coords(G, u, v, attrs)
        xs = [coord[0] for coord in coords]
        ys = [coord[1] for coord in coords]
        ax.plot(xs, ys, color=edge_color, linewidth=0.85, alpha=0.88, solid_capstyle="round")
        draw_direction_arrow(ax, coords, edge_color, linewidth=0.85, alpha=0.88, zorder=4)
    if G.nodes:
        pts = np.asarray([node_xy(G, node) for node in G.nodes()], dtype=float)
        ax.scatter(pts[:, 0], pts[:, 1], s=8, c=node_color, linewidths=0, zorder=5)
    finalize_publication_axes(ax, title, graph_bounds(G))
    save_publication_figure(fig, save_path)


def plot_matched_graph_comparison(
    G_gt,
    G_pred,
    metrics,
    title,
    save_path,
    gt_edge_color="#0072B2",
    pred_edge_color="#D55E00",
    matched_gt_edge_color="#0072B2",
    matched_pred_edge_color="#D55E00",
    unmatched_gt_edge_color="#A9C8E6",
    unmatched_pred_edge_color="#F2BE92",
    matched_gt_node_color="#0072B2",
    matched_pred_node_color="#D55E00",
    unmatched_gt_node_color="#4C78A8",
    unmatched_pred_node_color="#B54A17",
    match_line_color="#7A7A7A",
    major_step=None,
    bounds_override=None,
):
    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)
    matched_pred_edges = metrics.get("matched_pred_edges", set())
    matched_gt_edges = metrics.get("matched_gt_edges", set())
    match_pred_to_gt = metrics.get("match_pred_to_gt", {})
    match_gt_to_pred = metrics.get("match_gt_to_pred", {})

    for u, v, attrs in G_gt.edges(data=True):
        coords = edge_coords(G_gt, u, v, attrs)
        xs = [coord[0] for coord in coords]
        ys = [coord[1] for coord in coords]
        is_matched = (u, v) in matched_gt_edges
        line_color = matched_gt_edge_color if is_matched else unmatched_gt_edge_color
        line_width = 1.0 if is_matched else 0.5
        line_alpha = 0.92 if is_matched else 0.55
        line_zorder = 3 if is_matched else 1
        ax.plot(
            xs,
            ys,
            color=line_color,
            linewidth=line_width,
            alpha=line_alpha,
            zorder=line_zorder,
            solid_capstyle="round",
        )

    for u, v, attrs in G_pred.edges(data=True):
        coords = edge_coords(G_pred, u, v, attrs)
        xs = [coord[0] for coord in coords]
        ys = [coord[1] for coord in coords]
        is_matched = (u, v) in matched_pred_edges
        line_color = matched_pred_edge_color if is_matched else unmatched_pred_edge_color
        line_width = 1.0 if is_matched else 0.5
        line_alpha = 0.92 if is_matched else 0.55
        line_zorder = 4 if is_matched else 2
        ax.plot(
            xs,
            ys,
            color=line_color,
            linewidth=line_width,
            alpha=line_alpha,
            zorder=line_zorder,
            solid_capstyle="round",
        )

    gt_matched_nodes = [node for node in G_gt.nodes() if node in match_gt_to_pred]
    gt_unmatched_nodes = [node for node in G_gt.nodes() if node not in match_gt_to_pred]
    pred_matched_nodes = [node for node in G_pred.nodes() if node in match_pred_to_gt]
    pred_unmatched_nodes = [node for node in G_pred.nodes() if node not in match_pred_to_gt]

    if gt_matched_nodes:
        gt_pts = np.asarray([node_xy(G_gt, node) for node in gt_matched_nodes], dtype=float)
        ax.scatter(
            gt_pts[:, 0],
            gt_pts[:, 1],
            s=8,
            c=matched_gt_node_color,
            marker="o",
            linewidths=0,
            alpha=0.8,
            zorder=5,
        )
    if gt_unmatched_nodes:
        gt_pts = np.asarray([node_xy(G_gt, node) for node in gt_unmatched_nodes], dtype=float)
        ax.scatter(
            gt_pts[:, 0],
            gt_pts[:, 1],
            s=14,
            c=unmatched_gt_node_color,
            marker="s",
            linewidths=0,
            alpha=0.9,
            zorder=6,
        )
    if pred_matched_nodes:
        pred_pts = np.asarray([node_xy(G_pred, node) for node in pred_matched_nodes], dtype=float)
        ax.scatter(
            pred_pts[:, 0],
            pred_pts[:, 1],
            s=8,
            c=matched_pred_node_color,
            marker="o",
            linewidths=0,
            alpha=0.8,
            zorder=7,
        )
    if pred_unmatched_nodes:
        pred_pts = np.asarray([node_xy(G_pred, node) for node in pred_unmatched_nodes], dtype=float)
        ax.scatter(
            pred_pts[:, 0],
            pred_pts[:, 1],
            s=16,
            c=unmatched_pred_node_color,
            marker="x",
            linewidths=0.85,
            alpha=0.9,
            zorder=8,
        )

    for pred_node, gt_node in match_pred_to_gt.items():
        px, py = node_xy(G_pred, pred_node)
        gx, gy = node_xy(G_gt, gt_node)
        ax.plot([px, gx], [py, gy], linestyle="--", color=match_line_color, linewidth=0.45, alpha=0.18, zorder=2.5)

    legend_handles = [
        Line2D([0], [0], color=matched_gt_edge_color, lw=2, label="GT matched edge"),
        Line2D([0], [0], color=unmatched_gt_edge_color, lw=1, label="GT unmatched edge"),
        Line2D([0], [0], color=matched_pred_edge_color, lw=2, label="Ours matched edge"),
        Line2D([0], [0], color=unmatched_pred_edge_color, lw=1, label="Ours unmatched edge"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=matched_gt_node_color, markersize=6, label="GT matched node"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=unmatched_gt_node_color, markersize=6, label="GT unmatched node"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=matched_pred_node_color, markersize=6, label="Ours matched node"),
        Line2D([0], [0], marker="x", color=unmatched_pred_node_color, markersize=7, label="Ours unmatched node"),
        Line2D([0], [0], color=match_line_color, lw=1, ls="--", label="Node match link"),
    ]

    legend = ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        frameon=True,
        fontsize=LEGEND_FONTSIZE,
    )
    bounds = bounds_override if bounds_override is not None else graph_bounds(G_gt, G_pred)
    apply_grid_paper_axes(ax, title, bounds, legend, major_step=major_step)
    save_publication_figure(fig, save_path)


def print_metrics(title, metrics):
    print(f"\n{title}")
    print(f"Topo-Precision = {metrics['precision']:.4f}")
    print(f"Topo-Recall    = {metrics['recall']:.4f}")
    print(f"Topo-F1        = {metrics['f1']:.4f}")
    print(f"APLS           = {metrics['apls']:.4f}")
    print(f"APLS-Raw       = {metrics['raw_apls']:.4f}")
    print(f"APLS-Coverage  = {metrics['coverage']:.4f}")
    print(f"Cnode          = {metrics['c_node']:.4f}")


def write_metrics_summary(metrics_by_region, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        "region,Topo-Precision,Topo-Recall,Topo-F1,Directed-APLS-like,Directed-APLS-like-Raw,APLS-Coverage,Cnode"
    ]
    for region_name, metrics in metrics_by_region.items():
        lines.append(
            ",".join(
                [
                    region_name,
                    f"{metrics['precision']:.6f}",
                    f"{metrics['recall']:.6f}",
                    f"{metrics['f1']:.6f}",
                    f"{metrics['apls']:.6f}",
                    f"{metrics['raw_apls']:.6f}",
                    f"{metrics['coverage']:.6f}",
                    f"{metrics['c_node']:.6f}",
                ]
            )
        )

    (output_dir / "metrics_summary.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_dir / "metrics_summary.txt").write_text(
        "\n".join(
            [
                f"{region}\n"
                f"Topo-Precision = {metrics['precision']:.4f}\n"
                f"Topo-Recall    = {metrics['recall']:.4f}\n"
                f"Topo-F1        = {metrics['f1']:.4f}\n"
                f"Directed APLS-like = {metrics['apls']:.4f}\n"
                f"Directed APLS-like-Raw = {metrics['raw_apls']:.4f}\n"
                f"APLS-Coverage  = {metrics['coverage']:.4f}\n"
                f"Cnode          = {metrics['c_node']:.4f}\n"
                for region, metrics in metrics_by_region.items()
            ]
        ),
        encoding="utf-8",
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/4] 读取地图...")
    gt_raw_gdf = read_metric_gdf(GT_SHP_PATH)
    if has_valid_node_assignments(gt_raw_gdf):
        gt_gdf = gt_raw_gdf
        print(f"    - GT 已包含 source/target: {GT_SHP_PATH}")
    else:
        gt_gdf, gt_path = annotate_lines_with_nodes(
            GT_SHP_PATH,
            GT_WITH_NODES_PATH,
            feature_prefix="gt",
        )
        print(f"    - GT 已补 source/target 并保存: {gt_path}")
    pred_raw_gdf = read_metric_gdf(PRED_SHP_PATH)
    if has_valid_node_assignments(pred_raw_gdf):
        pred_gdf = pred_raw_gdf
        print(f"    - Pred 已包含有效 source/target: {PRED_SHP_PATH}")
    else:
        pred_gdf, pred_path = annotate_lines_with_nodes(
            PRED_SHP_PATH,
            PRED_WITH_NODES_PATH,
            feature_prefix="pred",
        )
        print(f"    - Pred 已补 source/target 并保存: {pred_path}")

    print("[2/4] 构建原生拓扑骨架...")
    G_gt, G_gt_raw, _ = build_native_topology(gt_gdf, "GT")
    G_pred, G_pred_raw, _ = build_native_topology(pred_gdf, "Pred")
    G_gt_junction = junction_subgraph(G_gt)
    G_pred_junction = junction_subgraph(G_pred)
    G_gt_lane_change = lane_change_subgraph(G_gt)
    G_pred_lane_change = lane_change_subgraph(G_pred)
    G_gt_road_segment = road_segment_subgraph(G_gt)
    G_pred_road_segment = road_segment_subgraph(G_pred)
    overall_plot_bounds = graph_bounds(G_gt, G_pred)
    shared_grid_step = compute_reference_grid_step(overall_plot_bounds)

    print(f"真值骨架: {G_gt.number_of_nodes()} 节点, {G_gt.number_of_edges()} 边")
    print(f"预测骨架: {G_pred.number_of_nodes()} 节点, {G_pred.number_of_edges()} 边")
    print(f"真值原始有向图: {G_gt_raw.number_of_nodes()} 节点, {G_gt_raw.number_of_edges()} 边")
    print(f"预测原始有向图: {G_pred_raw.number_of_nodes()} 节点, {G_pred_raw.number_of_edges()} 边")
    print(f"真值路口子图: {G_gt_junction.number_of_nodes()} 节点, {G_gt_junction.number_of_edges()} 边")
    print(f"预测路口子图: {G_pred_junction.number_of_nodes()} 节点, {G_pred_junction.number_of_edges()} 边")
    print(f"真值变道子图: {G_gt_lane_change.number_of_nodes()} 节点, {G_gt_lane_change.number_of_edges()} 边")
    print(f"预测变道子图: {G_pred_lane_change.number_of_nodes()} 节点, {G_pred_lane_change.number_of_edges()} 边")
    print(f"真值道路段子图: {G_gt_road_segment.number_of_nodes()} 节点, {G_gt_road_segment.number_of_edges()} 边")
    print(f"预测道路段子图: {G_pred_road_segment.number_of_nodes()} 节点, {G_pred_road_segment.number_of_edges()} 边")

    print("[3/4] 输出骨架可视化...")
    plot_graph(G_gt_raw, "GT Raw Directed Graph", OUTPUT_DIR / "gt_raw_directed_graph.png", "#8DAA91", "#315C48")
    plot_graph(G_pred_raw, "Pred Raw Directed Graph", OUTPUT_DIR / "pred_raw_directed_graph.png", "#8AA7C2", "#1F4E79")
    plot_graph(G_gt, "GT Native Topology", OUTPUT_DIR / "gt_native_topology.png", "#49796B", "#23483D")
    plot_graph(G_pred, "Pred Native Topology", OUTPUT_DIR / "pred_native_topology.png")
    plot_graph(
        G_gt_junction,
        "GT Junction Topology",
        OUTPUT_DIR / "gt_junction_topology.png",
        "#49796B",
        "#23483D",
    )
    plot_graph(G_pred_junction, "Pred Junction Topology", OUTPUT_DIR / "pred_junction_topology.png")
    plot_graph(
        G_gt_lane_change,
        "GT Lane Change Topology",
        OUTPUT_DIR / "gt_lane_change_topology.png",
        "#A26D3D",
        "#6F431A",
    )
    plot_graph(
        G_pred_lane_change,
        "Pred Lane Change Topology",
        OUTPUT_DIR / "pred_lane_change_topology.png",
        "#C27B48",
        "#8E4F1D",
    )
    plot_graph(
        G_gt_road_segment,
        "GT Road Segment Topology",
        OUTPUT_DIR / "gt_road_segment_topology.png",
        "#4C6A92",
        "#274060",
    )
    plot_graph(
        G_pred_road_segment,
        "Pred Road Segment Topology",
        OUTPUT_DIR / "pred_road_segment_topology.png",
        "#7A8E3A",
        "#4F6228",
    )

    print("[4/4] 计算整体和局部拓扑指标...")
    overall = evaluate_directed_topology(G_gt, G_pred, tau=MATCH_TOLERANCE)
    junction = evaluate_directed_topology(G_gt_junction, G_pred_junction, tau=MATCH_TOLERANCE)
    lane_change = evaluate_directed_topology(G_gt_lane_change, G_pred_lane_change, tau=MATCH_TOLERANCE)
    road_segment = evaluate_directed_topology(G_gt_road_segment, G_pred_road_segment, tau=MATCH_TOLERANCE)

    print("[5/5] 输出匹配对比图...")
    plot_matched_graph_comparison(
        G_gt,
        G_pred,
        overall,
        "GT vs Pred Topology Match",
        OUTPUT_DIR / "overall_topology_match.png",
        major_step=shared_grid_step,
        bounds_override=overall_plot_bounds,
    )
    plot_matched_graph_comparison(
        G_gt_junction,
        G_pred_junction,
        junction,
        "GT vs Pred Junction Match",
        OUTPUT_DIR / "junction_topology_match.png",
        major_step=shared_grid_step,
        bounds_override=overall_plot_bounds,
    )
    plot_matched_graph_comparison(
        G_gt_lane_change,
        G_pred_lane_change,
        lane_change,
        "GT vs Pred Lane Change Match",
        OUTPUT_DIR / "lane_change_topology_match.png",
        major_step=shared_grid_step,
        bounds_override=overall_plot_bounds,
    )
    plot_matched_graph_comparison(
        G_gt_road_segment,
        G_pred_road_segment,
        road_segment,
        "GT vs Pred Road Segment Match",
        OUTPUT_DIR / "road_segment_topology_match.png",
        major_step=shared_grid_step,
        bounds_override=overall_plot_bounds,
    )
    plot_endpoint_matched_but_edge_unmatched(
        G_gt,
        G_pred,
        overall,
        "Overall Endpoint-Matched but Edge-Unmatched",
        OUTPUT_DIR / "overall_edge_mismatch_only.png",
        major_step=shared_grid_step,
        bounds_override=overall_plot_bounds,
    )
    plot_endpoint_matched_but_edge_unmatched(
        G_gt_junction,
        G_pred_junction,
        junction,
        "Junction Endpoint-Matched but Edge-Unmatched",
        OUTPUT_DIR / "junction_edge_mismatch_only.png",
        major_step=shared_grid_step,
        bounds_override=overall_plot_bounds,
    )
    plot_endpoint_matched_but_edge_unmatched(
        G_gt_lane_change,
        G_pred_lane_change,
        lane_change,
        "Lane Change Endpoint-Matched but Edge-Unmatched",
        OUTPUT_DIR / "lane_change_edge_mismatch_only.png",
        major_step=shared_grid_step,
        bounds_override=overall_plot_bounds,
    )
    plot_endpoint_matched_but_edge_unmatched(
        G_gt_road_segment,
        G_pred_road_segment,
        road_segment,
        "Road Segment Endpoint-Matched but Edge-Unmatched",
        OUTPUT_DIR / "road_segment_edge_mismatch_only.png",
        major_step=shared_grid_step,
        bounds_override=overall_plot_bounds,
    )
    plot_unmatched_nodes_with_matched_neighbors(
        G_gt,
        G_pred,
        overall,
        "Overall Unmatched Nodes with Matched Neighbors",
        OUTPUT_DIR / "overall_unmatched_node_candidates.png",
        major_step=shared_grid_step,
        bounds_override=overall_plot_bounds,
    )
    plot_unmatched_nodes_with_matched_neighbors(
        G_gt_junction,
        G_pred_junction,
        junction,
        "Junction Unmatched Nodes with Matched Neighbors",
        OUTPUT_DIR / "junction_unmatched_node_candidates.png",
        major_step=shared_grid_step,
        bounds_override=overall_plot_bounds,
    )
    plot_unmatched_nodes_with_matched_neighbors(
        G_gt_lane_change,
        G_pred_lane_change,
        lane_change,
        "Lane Change Unmatched Nodes with Matched Neighbors",
        OUTPUT_DIR / "lane_change_unmatched_node_candidates.png",
        major_step=shared_grid_step,
        bounds_override=overall_plot_bounds,
    )
    plot_unmatched_nodes_with_matched_neighbors(
        G_gt_road_segment,
        G_pred_road_segment,
        road_segment,
        "Road Segment Unmatched Nodes with Matched Neighbors",
        OUTPUT_DIR / "road_segment_unmatched_node_candidates.png",
        major_step=shared_grid_step,
        bounds_override=overall_plot_bounds,
    )
    write_edge_debug_report(
        "overall",
        G_gt,
        G_pred,
        overall,
        DEBUG_DIR / "overall_edge_debug.txt",
    )
    write_edge_debug_report(
        "junction",
        G_gt_junction,
        G_pred_junction,
        junction,
        DEBUG_DIR / "junction_edge_debug.txt",
    )
    write_edge_debug_report(
        "lane_change",
        G_gt_lane_change,
        G_pred_lane_change,
        lane_change,
        DEBUG_DIR / "lane_change_edge_debug.txt",
    )
    write_edge_debug_report(
        "road_segment",
        G_gt_road_segment,
        G_pred_road_segment,
        road_segment,
        DEBUG_DIR / "road_segment_edge_debug.txt",
    )
    write_unmatched_node_debug_report(
        "overall",
        G_gt,
        G_pred,
        overall,
        DEBUG_DIR / "overall_unmatched_node_debug.txt",
    )
    write_unmatched_node_debug_report(
        "junction",
        G_gt_junction,
        G_pred_junction,
        junction,
        DEBUG_DIR / "junction_unmatched_node_debug.txt",
    )
    write_unmatched_node_debug_report(
        "lane_change",
        G_gt_lane_change,
        G_pred_lane_change,
        lane_change,
        DEBUG_DIR / "lane_change_unmatched_node_debug.txt",
    )
    write_unmatched_node_debug_report(
        "road_segment",
        G_gt_road_segment,
        G_pred_road_segment,
        road_segment,
        DEBUG_DIR / "road_segment_unmatched_node_debug.txt",
    )

    write_metrics_summary(
        {
            "overall": overall,
            "junction": junction,
            "lane_change": lane_change,
            "road_segment": road_segment,
        },
        OUTPUT_DIR,
    )

    print_metrics("整体拓扑", overall)
    print_metrics("局部路口拓扑", junction)
    print_metrics("局部变道拓扑", lane_change)
    print_metrics("道路段拓扑", road_segment)


if __name__ == "__main__":
    main()
