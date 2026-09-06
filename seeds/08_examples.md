# Patterns that work

Fix every source of randomness and pin the estimator so repeated runs agree:

```python
np.random.seed(0)
clf = LogisticRegression(C=1.0, max_iter=500, random_state=0)
```

Build features with explicit handling of missing values rather than letting them propagate:

```python
x = pd.to_numeric(df["some_column"], errors="coerce")
has = x.notna().astype(float)
x = x.fillna(x.median() if x.notna().any() else 0.0)
```

Clip the final output away from the boundaries so a confident miss is survivable:

```python
return np.clip(p.astype(np.float64), 0.02, 0.98)
```

Compute sequential statistics in date order and freeze them at the end of the training rows; never update them with anything from the rows you are predicting.
