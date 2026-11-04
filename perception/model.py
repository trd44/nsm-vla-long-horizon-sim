"""Geometry-residual gradient boosting: pixels and EEF only, sklearn-style per axis.

reg_z predicts height as a correction to the wrist/agent triangulated height.
reg_x and reg_y evaluate the agent camera ray at that predicted height and learn
the residual, so a height error moves the estimate along the ray instead of off it.
Each estimator exposes n_features_in_ == 14 and predict(X), as the executor expects.
"""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from perception.features import engineer, ray_xy

# Engineered columns the x/y heads may use when wrist boxes are excluded: class, agent
# box, EEF, agent ray, has_wrist, agent box shape. Wrist ray noise from unobserved
# orientation can exceed agent pixel noise, so wrist then informs x/y only through z.
AGENT_COLS = [0, 1, 2, 3, 4, 5, 11, 12, 13, 14, 15, 16, 17, 18, 19, 22, 34, 35, 36]
DEFAULT_PARAMS = dict(learning_rate=0.05, max_iter=600, max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=1.0)


def make_gb(params, seed=0):
    return HistGradientBoostingRegressor(early_stopping=False, random_state=seed, **params)


class ZHead(BaseEstimator, RegressorMixin):
    n_features_in_ = 14

    def __init__(self, gb=None):
        self.gb = gb

    def predict(self, X):
        z_geo, _, f = engineer(X)
        return z_geo + self.gb.predict(f)


class RayHead(BaseEstimator, RegressorMixin):
    n_features_in_ = 14

    def __init__(self, axis=0, z_head=None, gb=None, agent_only=False):
        self.axis, self.z_head, self.gb, self.agent_only = axis, z_head, gb, agent_only

    def predict(self, X):
        z = self.z_head.predict(X)
        _, slope, f = engineer(X)
        f = f[:, AGENT_COLS] if self.agent_only else f
        return ray_xy(slope, z)[:, self.axis] + self.gb.predict(np.column_stack([f, z]))


def fit_ray_cascade(X, Y, groups, params=None, n_folds=3, seed=0, agent_only=False, sample_weight=None):
    """Fit {reg_x, reg_y, reg_z}. X/Y head residuals train on out-of-fold z predictions."""
    params = {**DEFAULT_PARAMS, **(params or {})}
    z_geo, slope, f = engineer(X)
    w = np.ones(len(X)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    z_oof = np.zeros(len(X))
    for tr, te in GroupKFold(n_folds).split(X, groups=groups):
        gb = make_gb(params, seed).fit(f[tr], Y[tr, 2] - z_geo[tr], sample_weight=w[tr])
        z_oof[te] = z_geo[te] + gb.predict(f[te])
    z_head = ZHead(make_gb(params, seed).fit(f, Y[:, 2] - z_geo, sample_weight=w))
    base = ray_xy(slope, z_oof)
    f2 = np.column_stack([f[:, AGENT_COLS] if agent_only else f, z_oof])
    out = {"reg_z": z_head}
    for axis, name in enumerate(("reg_x", "reg_y")):
        gb = make_gb(params, seed).fit(f2, Y[:, axis] - base[:, axis], sample_weight=w)
        out[name] = RayHead(axis, z_head, gb, agent_only)
    return out


def fit_plain(X, Y, params=None, seed=0):
    """Plain per-axis gradient boosting on the raw 14 features."""
    params = {**DEFAULT_PARAMS, **(params or {})}
    return {n: make_gb(params, seed).fit(X, Y[:, i]) for i, n in enumerate(("reg_x", "reg_y", "reg_z"))}


def predict_xyz(model, X):
    return np.column_stack([model[k].predict(X) for k in ("reg_x", "reg_y", "reg_z")])
