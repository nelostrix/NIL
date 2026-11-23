# ranking

Query-safe native learning-to-rank and held-out query metrics.

## Usage

```python
import nil
print(nil.ranking.evaluate([3, 0, 1, 0], [.9, .1, .8, .2], ['a', 'a', 'b', 'b'], k=2))
# model = nil.ranking.fit(X_train, relevance, query_ids)
# scores = model.predict(X_test)
```

## Parameters

fit(features, relevance, groups, estimator=None, sample_weight=None). Default LGBMRanker; supplied estimators must accept group-size training metadata. split uses whole-query group splitting. evaluate(targets, scores, groups, k=10).

## Output

Serializable query_ranker and mean per-query NDCG@k/MAP@k. Input query rows may be interleaved; training groups are stably regrouped.

## Requirements

Metrics/splits: nil-sdk[classical]. Default training: nil-sdk[lightgbm].

## Limitations

Relevance grades for the native ranker are integers 0–31. Metrics use gain 2**relevance-1 and stable input-order score ties. MAP treats relevance>0 as positive and divides by min(total positives,k); no-positive queries score zero. Held-out query IDs must be disjoint from training queries; split before fitting preprocessing. This is not counterfactual/unbiased click ranking or arbitrary ranking-backend compatibility. Use nil.save/load for persistence.
