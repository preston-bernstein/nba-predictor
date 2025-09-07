# tests/test_metrics.py
import numpy as np
from src.model import metrics as metrics_mod

class _DummyModel:
    def __init__(self, bias=0.0):
        self.bias = bias
    def fit(self, X, y):
        return self
    def predict_proba(self, X):
        # simple sigmoid on sum(X)+bias
        s = 1 / (1 + np.exp(-(X.sum(axis=1) + self.bias)))
        return np.column_stack([1 - s, s])

def test_fit_and_score_handles_single_class():
    X_tr = np.array([[0,0],[1,1],[2,2]])
    y_tr = np.array([0,1,1])
    X_te = np.array([[0,0]])        # tiny, likely single class in y_te check path
    y_te = np.array([1])             # single class -> roc_auc should fallback to 0.5
    m = _DummyModel()
    out = metrics_mod.fit_and_score(m, X_tr, y_tr, X_te, y_te)
    assert set(["n_train","n_test","accuracy","roc_auc"]).issubset(out.keys())
    assert out["n_train"] == len(y_tr)
    assert out["n_test"] == len(y_te)
    assert 0.0 <= out["accuracy"] <= 1.0
    assert out["roc_auc"] == 0.5

def test_selection_key_orders_by_roc_then_acc():
    k1 = metrics_mod.selection_key({"roc_auc": 0.60, "accuracy": 0.70})
    k2 = metrics_mod.selection_key({"roc_auc": 0.65, "accuracy": 0.60})
    k3 = metrics_mod.selection_key({"roc_auc": float("nan"), "accuracy": 0.99})
    assert k2 > k1          # higher ROC wins over accuracy
    assert k1 > k3          # NaN ROC treated as -inf
