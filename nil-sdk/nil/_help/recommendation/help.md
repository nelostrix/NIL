# recommendation

Sparse item-cosine retrieval with bounded candidates and cold starts.

## Usage

```python
import nil
from scipy.sparse import csr_matrix
interactions = csr_matrix([[1, 1, 0], [1, 0, 1], [0, 1, 1]])
model = nil.recommendation.fit(interactions, neighbors=2)
print(model.recommend(0, k=1))
print(model.recommend(None, k=2))
```

## Parameters

fit(sparse_user_item_matrix, neighbors=20, max_items=10000, max_pairs=2000000). recommend(user=None,k=10,exclude_seen=True). evaluate(model,heldout,k=10,max_users=10000) requires disjoint train/heldout interactions.

## Output

Serializable item_cosine estimator; recommendations are (item_column_index, score) pairs. Held-out precision/recall/NDCG are averaged over users with positive held-out interactions.

## Requirements

nil-sdk[classical], including SciPy sparse arrays.

## Limitations

Implicit nonnegative interactions only; similarity uses binary item co-occurrence, user scores weight by supplied interaction magnitudes. Cold users (None or known empty rows) use popularity, as do rows with no neighbor signal. Unknown numeric user IDs fail; user/item external ID mapping is caller-owned. Sparse multiplication is preflighted with a conservative sum-of-squared-user-degrees budget; final neighbors are bounded. Item scores allocate at most max_items values, not a user×item dense matrix. No neural/ALS or approximate-nearest-neighbor engine. Stable item-index tie breaking; precision@k uses fixed k even if fewer items remain. Use nil.save/load.
