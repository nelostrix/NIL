# multinomial_nb

Native `sklearn.naive_bayes.MultinomialNB` adapter in classification.

## Usage

```python
import nil

model = nil.model("multinomial_nb", alpha=1.0)
X = [[0., 0.], [0.1, 0.2], [1., 1.], [1.1, 1.2], [0.2, 0.1], [1.2, 1.1]]
model.fit(X, [0, 0, 1, 1, 0, 1])
print(model.predict(X[:2]))
```

## Parameters

alpha smooths nonnegative count-like features. Inputs must not contain negative values.

## Output

A native MultinomialNB; fitting/prediction/transform methods retain backend behavior.

## Requirements

Install `nil-sdk[classical]` for sklearn operations. Named DataFrame inputs additionally require pandas. Backends load only when called.

## Limitations

Native model/data constraints apply. This tiny example establishes API usage, not useful quality. Tune only with held-out folds and explicit budgets; some algorithms can require substantial time or memory.
