# Social Network Analysis — Reddit Gaming Community Detection

A comparative study of community detection algorithms (Louvain, Leiden, Spectral Clustering) applied to a Reddit gaming social network built from January 2026 comments.

## Overview

This project constructs a weighted user interaction graph from Reddit gaming comments and evaluates three community detection algorithms using internal metrics (Modularity, Conductance), external metrics (NMI, ARI, Purity) against ground-truth subreddit labels, and LDA-based topic overlap analysis.

## Pipeline

```
Raw Reddit Comments → Graph Construction → Node Indexing
    → Community Detection (Louvain / Leiden / Spectral)
    → Internal Evaluation (Modularity, Conductance)
    → External Evaluation (NMI, ARI, Purity)
    → LDA Topic Modeling → LDA-based Comparison (DDS, MCR, DCR)
    → Gephi Export → Visualization
```

## Repository Structure

```
├── src/
│   ├── graph/
│   │   ├── graph_builder.py        # Build user interaction graph from Reddit comments
│   │   └── graph_indexing.py       # Map user string IDs to numeric indices
│   ├── communities/
│   │   ├── run_louvain.py          # Louvain community detection
│   │   ├── run_leiden.py           # Leiden community detection
│   │   └── run_spectral.py         # Spectral clustering
│   ├── evaluate/
│   │   ├── internal_metrics.py     # Modularity & Avg Conductance
│   │   └── external_metrics.py     # NMI, ARI, Purity
│   ├── LDA/
│   │   ├── real_community.py       # LDA on ground-truth subreddits
│   │   ├── algorithm_communites.py # LDA on detected communities
│   │   └── evaluate_lda.py         # DDS, MCR, DCR metrics
│   ├── export_gephi.py             # Export GEXF with community coloring
│   ├── export_gexf_centroid.py     # Centroid labels for Gephi graphs
│   └── count_community.py          # Count members per community
├── notebooks/
│   ├── algo_louvain.ipynb          # Louvain analysis
│   ├── algo_leiden.ipynb           # Leiden analysis
│   ├── algo_spectral.ipynb         # Spectral analysis
│   ├── compare_metrics/            # Radar & bar chart comparisons
│   └── compare_topic/              # LDA topic visualization
├── data/
│   └── graph/                      # edges.csv, nodes.csv (27K nodes, 155K edges)
├── outputs/
│   ├── communities/                # Detected community assignments
│   ├── count_communities/          # Community member counts
│   ├── gexf/                       # Gephi visualization files
│   ├── LDA/                        # Topic modeling results
│   └── metrics/                    # Evaluation scores
└── img/                            # Visualization snapshots
```

## Dataset

- **Source**: Reddit comments from January 2026, filtered for gaming subreddits
- **Nodes**: 27,147 unique users
- **Edges**: ~155,000 weighted interactions (reply-based)
- **Threshold**: Minimum 3 interactions to filter spam/low-activity users
- **Ground truth**: 27 gaming subreddits used as community labels

## Results

### Internal Metrics

| Algorithm | Modularity | Avg Conductance | Communities |
|-----------|-----------|-----------------|-------------|
| Louvain   | 0.8387    | 0.0587          | 35          |
| Leiden    | **0.8561**| 0.0654          | 44          |
| Spectral  | 0.5991    | **0.0248**      | 20          |

### External Metrics (vs. ground truth)

| Algorithm | NMI    | ARI    | Purity  |
|-----------|--------|--------|---------|
| Louvain   | 0.8236 | 0.6679 | 71.83%  |
| Leiden    |**0.8822**|**0.8138**|**83.70%**|
| Spectral  | 0.5440 | 0.1371 | 37.75%  |

### LDA Topic Overlap

| Algorithm | DDS   | MCR    | DCR   |
|-----------|-------|--------|-------|
| Louvain   |16.26% | 62.96% | 94.95%|
| Leiden    |15.87% |**77.78%**| 88.63%|
| Spectral  |14.84% | 22.22% |**100%**|

**Leiden** achieves the best balance of internal structure, ground-truth alignment, and topic coverage.

## Setup

```bash
# Create conda environment
conda create -n social-network python=3.14
conda activate social-network

# Install dependencies
pip install pandas numpy networkx scipy scikit-learn python-louvain igraph leidenalg

# Configure environment
# Edit .env to set PROJECT_ROOT to your local path
```

## Usage

```bash
# 1. Build graph
python src/graph/graph_builder.py
python src/graph/graph_indexing.py

# 2. Detect communities
python src/communities/run_louvain.py
python src/communities/run_leiden.py
python src/communities/run_spectral.py

# 3. Evaluate
python src/evaluate/internal_metrics.py
python src/evaluate/external_metrics.py
python src/count_community.py

# 4. Topic modeling
python src/LDA/real_community.py
python src/LDA/algorithm_communites.py
python src/LDA/evaluate_lda.py

# 5. Export for visualization
python src/export_gephi.py
# Load into Gephi, run ForceAtlas2, save as *_pos.gexf
python src/export_gexf_centroid.py
```

## Visualization

GEXF files in `outputs/gexf/` are designed for [Gephi](https://gephi.org/). Apply ForceAtlas2 layout and use the exported centroid files for labeled community visualization.

## Built With

- [NetworkX](https://networkx.org/) — graph construction and analysis
- [igraph](https://igraph.org/) + [leidenalg](https://github.com/vtraag/leidenalg) — Leiden algorithm
- [python-louvain](https://github.com/taynaud/python-louvain) — Louvain algorithm
- [scikit-learn](https://scikit-learn.org/) — Spectral Clustering, LDA, external metrics
- [Pandas](https://pandas.pydata.org/) / [NumPy](https://numpy.org/) — data processing
- [Gephi](https://gephi.org/) — network visualization

## License

MIT

## After Update — Heatmap Format Improvement

The heatmap in `notebooks/compare_topic/heatmap.ipynb` was updated to sort both rows and columns by a **fixed ordering rule**, creating a stair-like diagonal pattern where high scores concentrate along the diagonal from top-left to bottom-right.

### Sorting Logic

**Columns (X-axis) — fixed order:**
- Sorted by Community ID in ascending order: **Community 0 → Community 21**
- This ensures communities appear left-to-right in a consistent, predictable sequence

**Rows (Y-axis) — best-match order:**
- For each community column in order (0 → 21), find the **subreddit with the highest topic overlap score** that hasn't been placed yet
- Place that subreddit at the corresponding row position, aligning the best-matching subreddit with its community
- Any remaining unmatched subreddits are appended at the bottom

### Result

| Aspect | Before | After |
|--------|--------|-------|
| **Columns** | Unsorted, labeled `Cluster {id}` | Fixed ascending order, labeled `Community {id}` |
| **Rows** | Original subreddit order (alphabetical) | Each community's best-matching subreddit placed at its diagonal position |
| **Visual** | Scattered high scores | Descending diagonal of high scores (stair-like) |

This makes it immediately visible which subreddit each algorithm community best corresponds to — the diagonal cells show the strongest matches, with values decreasing naturally as you move away from the diagonal.
