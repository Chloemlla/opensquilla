#!/usr/bin/env python3
"""One-off offline converter: SquillaRouter V4 Phase 3 trained assets -> plain JSON.

Reads the sklearn/lightgbm pickles under the bundle root and writes the
plain-data JSON files that the Rust loaders consume. The Rust side never reads
.pkl/.joblib/.bin; this script is the bridge. It is NOT run inside the Rust
build.

Usage:
    python export_router_params.py [ROOT]

ROOT defaults to the v4.2_phase3_inference bundle directory. Output goes to
ROOT/params/*.json plus ROOT/artifact_manifest.json.
"""

import json
import pathlib
import sys

DEFAULT_ROOT = pathlib.Path(
    r"F:\Repositories\GitHub\opensquilla\src\opensquilla\squilla_router\models\v4.2_phase3_inference"
)

# Heavy optional deps are wrapped so a missing one produces a clear message
# instead of an import-time traceback. Only the fitted attributes are touched;
# the sklearn estimators themselves are never imported or instantiated.
try:
    import numpy as np  # noqa: F401
except ImportError:
    np = None

try:
    import scipy.sparse as sp_sparse
except ImportError:
    sp_sparse = None

try:
    import joblib
except ImportError:
    joblib = None

try:
    import lightgbm
except ImportError:
    lightgbm = None


def write_json(obj, path):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)
    print(f"wrote {path}")


def require(module, name):
    if module is None:
        raise SystemExit(
            f"error: missing optional dependency '{name}'. "
            f"Install it (e.g. `pip install {name}`) before running this script."
        )
    return module


def export_tfidf(path, out_path):
    require(joblib, "joblib")
    vec = joblib.load(str(path))
    if not hasattr(vec, "vocabulary_") or not hasattr(vec, "idf_"):
        raise ValueError(f"{path} is not a fitted TfidfVectorizer")
    ngram = vec.ngram_range
    # vocabulary_ keys are sklearn's raw char_wb n-gram strings (including
    # padding spaces, e.g. " ab"); dump them verbatim, do NOT normalize.
    write_json(
        {
            "ngram_range": [int(ngram[0]), int(ngram[1])],
            "sublinear_tf": bool(vec.sublinear_tf),
            "max_features": (
                int(vec.max_features)
                if vec.max_features is not None
                else 10000  # documented training cap; Rust field is non-null usize
            ),
            "vocabulary": {str(k): int(v) for k, v in vec.vocabulary_.items()},
            "idf": [float(x) for x in vec.idf_],
        },
        out_path,
    )


def export_svd(path, out_path):
    require(joblib, "joblib")
    sp = require(sp_sparse, "scipy")
    svd = joblib.load(str(path))
    if not hasattr(svd, "components_"):
        raise ValueError(f"{path} is not a fitted TruncatedSVD")
    coo = sp.coo_matrix(svd.components_)
    write_json(
        {
            "n_components": int(coo.shape[0]),
            "n_features": int(coo.shape[1]),
            "rows": [int(r) for r in coo.row],
            "cols": [int(c) for c in coo.col],
            "values": [float(d) for d in coo.data],
        },
        out_path,
    )


def extract_pca(obj):
    """Return the sklearn PCA from either a bare PCA or a wrapping dict."""
    if isinstance(obj, dict):
        for key in ("pca", "pca_", "PCA"):
            val = obj.get(key)
            if val is not None and hasattr(val, "components_"):
                return val
        for val in obj.values():
            if hasattr(val, "components_"):
                return val
        raise ValueError("object is a dict but none of its values looks like a fitted PCA")
    return obj


def export_bge_pca(path, out_path):
    require(joblib, "joblib")
    obj = joblib.load(str(path))
    pca = extract_pca(obj)
    if not hasattr(pca, "components_") or not hasattr(pca, "mean_"):
        raise ValueError(f"{path} does not contain a fitted PCA")
    write_json(
        {
            "components": [[float(x) for x in row] for row in pca.components_],
            "mean": [float(x) for x in pca.mean_],
        },
        out_path,
    )


def export_scaler(path, out_path):
    require(joblib, "joblib")
    scaler = joblib.load(str(path))
    if not hasattr(scaler, "mean_") or not hasattr(scaler, "scale_"):
        raise ValueError(f"{path} is not a fitted StandardScaler")
    write_json(
        {
            "mean": [float(x) for x in scaler.mean_],
            "scale": [float(x) for x in scaler.scale_],
        },
        out_path,
    )


def default_left(node):
    """LightGBM numeric trees default missing values to the left child."""
    if "default_left" in node and node["default_left"] is not None:
        return bool(node["default_left"])
    # "None" missing_type -> missing goes left; "Zero"/absent also left.
    return True


def flatten_tree(tree_structure):
    """Flatten one LightGBM tree by pre-order traversal into arrays in a
    unified node-id space. Every array has length == num_nodes. Each node's six
    entries live at the SAME index (its pre-order id): internal nodes reserve
    their split slots at the current index, recurse into children (which fill
    the following indices), then back-fill the child ids into the reserved
    slots. Leaf nodes carry -1 sentinels and the per-class leaf_value."""
    split_feature = []
    split_threshold = []
    left_child = []
    right_child = []
    default_left = []
    leaf_value = []

    def visit(node):
        # The next pre-order id equals the current length of the arrays.
        i = len(split_feature)
        if "leaf_value" in node:
            split_feature.append(-1)
            split_threshold.append(0.0)
            left_child.append(-1)
            right_child.append(-1)
            default_left.append(False)
            lv = node["leaf_value"]
            if isinstance(lv, (list, tuple)):
                leaf_value.append([float(x) for x in lv])
            else:
                leaf_value.append([float(lv)])
            return i

        split_feature.append(int(node["split_feature"]))
        split_threshold.append(float(node["threshold"]))
        # Reserve this node's child slots now so all six arrays stay aligned
        # at index i; the ids are back-filled after the subtrees are appended.
        left_child.append(-1)
        right_child.append(-1)
        default_left.append(default_left(node))
        leaf_value.append([0.0, 0.0, 0.0, 0.0])
        left_child[i] = visit(node["left_child"])
        right_child[i] = visit(node["right_child"])
        return i

    visit(tree_structure)

    return {
        "split_feature": split_feature,
        "split_threshold": split_threshold,
        "left_child": left_child,
        "right_child": right_child,
        "default_left": default_left,
        "leaf_value": leaf_value,
    }


def export_lgbm(path, out_path, num_class=4):
    require(lightgbm, "lightgbm")
    booster = lightgbm.Booster(model_file=str(path))
    model = booster.dump_model()
    trees = [flatten_tree(tree["tree_structure"]) for tree in model["tree_info"]]
    write_json(
        {
            "num_class": num_class,
            "num_iterations": len(trees),
            "trees": trees,
        },
        out_path,
    )


def main():
    root = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_ROOT
    root = root.resolve()
    if not root.is_dir():
        raise SystemExit(f"error: bundle root not found: {root}")

    params_dir = root / "params"
    params_dir.mkdir(parents=True, exist_ok=True)

    export_tfidf(root / "features" / "tfidf.pkl", params_dir / "tfidf.json")
    export_svd(root / "features" / "svd.pkl", params_dir / "svd.json")
    export_bge_pca(root / "features" / "bge_pca.joblib", params_dir / "bge_pca.json")
    export_scaler(root / "mlp" / "scaler.joblib", params_dir / "scaler.json")
    export_lgbm(root / "lgbm_main.bin", params_dir / "lgbm_main.json")

    params = {
        "tfidf": "tfidf.json",
        "svd": "svd.json",
        "bge_pca": "bge_pca.json",
        "scaler": "scaler.json",
        "lgbm_main": "lgbm_main.json",
    }

    aux_path = root / "lgbm_aux.bin"
    if aux_path.exists():
        export_lgbm(aux_path, params_dir / "lgbm_aux.json")
        params["lgbm_aux"] = "lgbm_aux.json"
    else:
        print(f"warning: {aux_path} not found; omitting lgbm_aux from manifest", file=sys.stderr)

    write_json(
        {
            "feature_dim": 390,
            "mlp_input_dim": 1536,
            "params": params,
        },
        root / "artifact_manifest.json",
    )


if __name__ == "__main__":
    main()
