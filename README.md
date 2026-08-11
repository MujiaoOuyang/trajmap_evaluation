# TrajMap Evaluation

Evaluation code for **TrajMap: Trajectory-Guided Topology-Preserving Global Lightweight HD Map Generation**.

This repository provides evaluation tools for assessing the **geometric accuracy** and **topological consistency** of generated vectorized HD maps.

Both prediction and ground-truth maps are represented in Shapefile (`.shp`) format.

---

## Evaluation Metrics

The evaluation mainly contains three parts:

### 1. Geometry Evaluation

Geometry evaluation is performed separately for lane lines and arrow markings.

#### Lane Lines

Predicted and ground-truth lane lines are uniformly sampled into point sets. Bidirectional nearest-neighbor distances are used to evaluate geometric accuracy.

The reported metrics include:

* Mean error
* Median error
* P90 error
* P95 error
* `Cov@r`: geometric coverage under distance threshold (r)
* `Cov-hard`: average coverage at 0.2 m, 0.5 m and 1.0 m
* `Cov-easy`: average coverage at 0.5 m, 1.0 m and 1.5 m

Lower distance errors indicate better geometric accuracy, while higher coverage indicates better geometric consistency.

#### Arrow Markings

Arrow elements are evaluated using:

* Matched arrow ratio
* Intersection over Union (IoU)
* Center localization error

---

### 2. Local Topology Evaluation

Local topology evaluates whether the generated map correctly preserves directed connectivity between nearby lane elements.

The reported metrics include:

* Topology Precision
* Topology Recall
* Topology F1-score
* Directed APLS at 30 m, 50 m and 80 m
* Break Rate (BR)
* Duplicate Rate (DR)

Higher Precision, Recall, F1 and APLS indicate better topology preservation.

Lower BR and DR indicate fewer broken or redundant lane structures.

---

### 3. Global Topology Evaluation

For global topology evaluation, the lane-level graph is compressed into a topology skeleton containing topology-critical nodes, such as:

* Junction nodes
* Lane-count-transition nodes
* Directed-chain boundary nodes

The reported metrics include:

* Topology Precision
* Topology Recall
* Topology F1-score
* Directed APLS-like score
* Node matching coverage

The final Directed APLS-like score jointly considers path consistency and topology-node matching coverage.

---

## Repository Structure

```text
trajmap_evaluation/
├── README.md
└── evaluation_trajmap/
    ├── geometry/
    │   ├── local/
    │   └── global/
    └── topology/
```

The repository currently contains evaluation scripts for:

* lane-line geometry;
* arrow geometry;
* local road-segment topology;
* global compressed topology.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/MujiaoOuyang/trajmap_evaluation.git
cd trajmap_evaluation
```

Install the main dependencies:

```bash
pip install numpy scipy geopandas shapely pyproj networkx matplotlib
```

---

## Input Data

Both the predicted map and ground-truth map should be provided in Shapefile format.

Example:

```text
ground_truth.shp
prediction.shp
```

The corresponding coordinate reference system (CRS) should be correctly defined so that geometric errors can be evaluated in meters.

Depending on the evaluation task, Shapefile attributes may also contain information such as:

```text
type
lane_type
source
target
```

---

## Usage

### Geometry Evaluation

```bash
python evaluation_trajmap/geometry/local/evaluate_vector_geometry.py \
    --gt path/to/ground_truth.shp \
    --pred path/to/prediction.shp
```

### Local Topology Evaluation

```bash
python evaluation_trajmap/topology/evaluate_topology_metrics.py \
    --gt path/to/ground_truth.shp \
    --pred path/to/prediction.shp
```

Please refer to the corresponding scripts for additional parameters and configuration options.

---

## Output

Evaluation results are exported in formats such as:

```text
.csv
.json
.png
```

The output contains quantitative evaluation results and, where applicable, visualization results for topology matching and debugging.

---

## Metric Summary

| Category        | Metrics                                   |
| --------------- | ----------------------------------------- |
| Lane Geometry   | Mean, Median, P90, P95, Cov@r             |
| Arrow Geometry  | Matching Ratio, IoU, Center Error         |
| Local Topology  | Precision, Recall, F1, APLS, BR, DR       |
| Global Topology | Precision, Recall, F1, Directed APLS-like |

Geometry and topology are evaluated separately because a map with good geometric alignment may still contain incorrect lane connectivity, and vice versa.

---

## Citation

If you find this evaluation toolkit useful in your research, please cite our TrajMap paper.

```bibtex
@article{trajmap,
  title   = {TrajMap: Trajectory-Guided Topology-Preserving Global Lightweight HD Map Generation},
  author  = {...},
  journal = {...},
  year    = {...}
}
```

The citation information will be updated after publication.

---

## Contact

For questions or suggestions, please open an issue in this repository.

