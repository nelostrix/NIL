# graphs

## Usage

```python
import nil
model = nil.graphs.gcn(4, 16, 2, task='node')
# logits = model(features, edge_index)
# nil.train(model, dataset, task=custom_masked_loss, device='cpu')
```

## Parameters

Finite float32/64 node features, int64 endpoints, node/edge/feature budgets; GCN dimensions and node/graph task; disjoint graph batching.

## Output

Normalized message-passing tensors, node/graph logits, pooled graph features, dtype-preserving immutable JSON graph artifacts.

## Requirements

Explicit Torch extra. No PyG/DGL/NetworkX runtime dependency.

## Limitations

Homogeneous graphs: base GCN is unweighted/undirected; weighted_gcn explicitly adds weights/direction. Caller owns train/test masks; transductive topology visibility differs from unseen-graph evaluation. Link splitting/scoring and bounded CPU sampling are available; temporal and distributed graph workflows remain open; heterogeneous node learning is described below.

See docs/SEVEN_WORKFLOWS.md for v1 acceptance and docs/ADVANCED_WORKFLOWS.md for this phase and remaining work.

## Weighted graphs and links

weighted_gcn and weighted_propagate accept explicit weights/direction. Duplicate edges sum; supply each undirected pair once. split_links excludes all known positives from true-nonedge samples. Use training positives alone for message passing. sample_neighbors is bounded CPU sampling, not an unbiased full-GCN estimate. Heterogeneous/temporal graphs remain open.

## Heterogeneous node learning

```python
relation = ('author', 'writes', 'paper')
model = nil.graphs.hetero_sage(
    {'author': 2, 'paper': 1}, 8, {'author': 2, 'paper': 2}, [relation])
# logits_by_type = model(features_by_type, {relation: int64_edge_index})
# path = nil.graphs.save_hetero(features_by_type, edges_by_relation, new_path)
# graph = nil.graphs.load_hetero(path)
```

Two typed self/relation-mean layers, ReLU between layers, separate parameters per relation, directed local endpoint indices. Duplicate edges count repeatedly; reverse relations are explicit. Missing relations raise; empty declared relations use [2,0] tensors. relation_mean has independent dense/gradient references. Checkpoints bind type/relation schemas; graph JSON preserves float32/64. Limits:16 types,64 relations,100k total nodes,1M features/edges/activations/parameters and hidden width512. Smaller projection/aggregation intermediates are selected automatically. Temporal graphs, typed batching/sampling and distributed graphs remain open. See docs/CONTINUED_WORKFLOWS.md.
