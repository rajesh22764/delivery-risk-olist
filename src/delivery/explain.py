"""Per-order explanations for the logistic regression: each feature's contribution to the log-odds, summed over its encoded columns."""
import numpy as np
import pandas as pd

from delivery.features import CATEGORICAL


def _origin(name: str, features: list) -> str:
    """Map an encoded column name such as 'cat__customer_state_SP' or 'heavy__distance_km' back to the original feature."""
    base = name.split("__", 1)[1]
    for f in sorted(features, key=len, reverse=True):
        if base == f or base.startswith(f + "_"):
            return f
    return base


def contributions(pipeline, X: pd.DataFrame, features: list) -> pd.DataFrame:
    """Rows: orders; columns: original features; values: contribution to the log-odds of being late (plus an 'intercept' column)."""
    pre, lr = pipeline.named_steps["pre"], pipeline.named_steps["lr"]
    Z = pre.transform(X[features])
    Z = Z.toarray() if hasattr(Z, "toarray") else np.asarray(Z)
    contrib = Z * lr.coef_.ravel()
    origins = [_origin(n, features) for n in pre.get_feature_names_out()]
    out = pd.DataFrame(contrib, columns=origins, index=X.index).T.groupby(level=0).sum().T
    out["intercept"] = float(lr.intercept_[0])
    return out


def coefficient_table(pipeline, features: list) -> pd.DataFrame:
    """Standardised coefficients per encoded column (numeric ones are on the same scale, so their sizes are comparable)."""
    pre, lr = pipeline.named_steps["pre"], pipeline.named_steps["lr"]
    names = pre.get_feature_names_out()
    return pd.DataFrame({"column": names, "feature": [_origin(n, features) for n in names], "coef": lr.coef_.ravel()}).sort_values("coef", key=np.abs, ascending=False)
