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

Homogeneous graphs: base GCN is unweighted/undirected; weighted_gcn explicitly adds weights/direction. Caller owns train/test masks; transductive topology visibility differs from unseen-graph evaluation. Link splitting/scoring and bounded CPU sampling are available; heterogeneous/temporal and distributed graph workflows remain open.

See docs/SEVEN_WORKFLOWS.md for v1 acceptance and docs/ADVANCED_WORKFLOWS.md for this phase and remaining work.

## Weighted graphs and links

weighted_gcn and weighted_propagate accept explicit weights/direction. Duplicate edges sum; supply each undirected pair once. split_links excludes all known positives from true-nonedge samples. Use training positives alone for message passing. sample_neighbors is bounded CPU sampling, not an unbiased full-GCN estimate. Heterogeneous/temporal graphs remain open.
