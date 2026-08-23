import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import OneClassSVM

try:
    import torch
except ImportError:
    torch = None


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_FILE = ROOT / "data" / "青花瓷色度统计LAB.xlsx"

SHEET_NAMES = ("元代", "明代")
SAMPLE_COL = "样品号"
EXPECTED_YUAN_ROWS = 310
EXPECTED_MING_ROWS = 25
TRAIN_SIZE = 285
YUAN_TEST_SIZE = 25
N_LAB_FEATURES = 6
EXPECTED_FEATURE_DIM = 32
FULL_MATRIX_CLASSICAL_METHODS = frozenset(("if", "lof", "ocsvm"))
REBUTTAL_ARRAY_METHODS = frozenset(
    (
        "deepsvdd",
        "abod",
        "mo_gaal",
        "so_gaal",
        "auto_encoder",
        "dif",
        "ecod",
        "slad",
    )
)
SUPPORTED_METHODS = (
    "if",
    "lof",
    "ocsvm",
    "deepsvdd",
    "abod",
    "mo_gaal",
    "so_gaal",
    "auto_encoder",
    "dif",
    "ecod",
    "slad",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lab-scale",
        required=True,
        choices=("raw", "standardized"),
        help="Required: source raw LAB or training-only standardized LAB.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=SUPPORTED_METHODS,
        default=["lof"],
        help="One or more source-baseline methods; default: lof.",
    )
    parser.add_argument("--data-file", type=Path, default=DEFAULT_DATA_FILE)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--seed",
        type=int,
        default=1235,
        help="Run seed; use 1235, 1236, and 12367 for stochastic methods.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Source baseline uses 100.",
    )
    parser.add_argument(
        "--classical-fit-epochs",
        type=int,
        default=1,
        help=(
            "Deprecated compatibility option; ignored. IF/LOF/OCSVM are "
            "fitted once on all 285 Yuan training rows."
        ),
    )
    parser.add_argument(
        "--no-roc-plot",
        action="store_true",
        help="Do not save per-method ROC figures.",
    )
    return parser.parse_args()


def set_global_seed(seed):
    """Seed every random source directly controlled by this script."""
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)


def load_legacy_features(data_file, lab_scale):
    """Reproduce source feature construction, changing only LAB scaling."""
    worksheets = pd.read_excel(data_file, sheet_name=list(SHEET_NAMES))
    yuan_raw = worksheets["元代"].copy()
    ming_raw = worksheets["明代"].copy()
    if len(yuan_raw) - 1 != EXPECTED_YUAN_ROWS:
        raise ValueError(
            f"Expected 310 Yuan physical rows, found {len(yuan_raw) - 1}"
        )
    if len(ming_raw) - 1 != EXPECTED_MING_ROWS:
        raise ValueError(
            f"Expected 25 Ming physical rows, found {len(ming_raw) - 1}"
        )

    lab_columns = list(yuan_raw.columns[3 : 3 + N_LAB_FEATURES])
    if len(lab_columns) != N_LAB_FEATURES:
        raise ValueError(f"Expected six LAB columns, found {lab_columns}")
    if list(ming_raw.columns[3 : 3 + N_LAB_FEATURES]) != lab_columns:
        raise ValueError("Yuan and Ming worksheets have different LAB columns")

    label_encoder = LabelEncoder()
    yuan_raw["青花呈色编码"] = label_encoder.fit_transform(yuan_raw["青花呈色"])
    yuan_encoded = pd.get_dummies(yuan_raw, columns=["器型"], prefix="器型")

    ming_raw["青花呈色编码"] = label_encoder.fit_transform(ming_raw["青花呈色"])
    ming_encoded = pd.get_dummies(ming_raw, columns=["器型"], prefix="器型")
    for column in yuan_encoded.columns:
        if column not in ming_encoded.columns:
            ming_encoded[column] = 0
    ming_encoded = ming_encoded[yuan_encoded.columns]

    yuan_features = np.asarray(yuan_encoded)[1:, 2:].astype(np.float32)
    ming_features = np.asarray(ming_encoded)[1:, 2:].astype(np.float32)
    if yuan_features.shape != (EXPECTED_YUAN_ROWS, EXPECTED_FEATURE_DIM):
        raise ValueError(f"Unexpected Yuan feature shape: {yuan_features.shape}")
    if ming_features.shape != (EXPECTED_MING_ROWS, EXPECTED_FEATURE_DIM):
        raise ValueError(f"Unexpected Ming feature shape: {ming_features.shape}")
    if not np.isfinite(yuan_features).all() or not np.isfinite(ming_features).all():
        raise ValueError("Feature matrix contains NaN or infinity")

    lab_scaler = None
    if lab_scale == "standardized":
        lab_scaler = StandardScaler().fit(
            yuan_features[:TRAIN_SIZE, :N_LAB_FEATURES]
        )
        yuan_features[:, :N_LAB_FEATURES] = lab_scaler.transform(
            yuan_features[:, :N_LAB_FEATURES]
        ).astype(np.float32)
        ming_features[:, :N_LAB_FEATURES] = lab_scaler.transform(
            ming_features[:, :N_LAB_FEATURES]
        ).astype(np.float32)
    elif lab_scale != "raw":
        raise ValueError(f"Unknown LAB scale: {lab_scale}")

    metadata = {
        "yuan_sample_ids": yuan_raw[SAMPLE_COL].iloc[1:].astype(str).to_numpy(),
        "ming_sample_ids": ming_raw[SAMPLE_COL].iloc[1:].astype(str).to_numpy(),
        "lab_columns": lab_columns,
        "encoded_feature_columns": [
            str(column) for column in list(yuan_encoded.columns[2:])
        ],
    }
    return yuan_features, ming_features, lab_scaler, metadata


def save_preprocessing(output_dir, lab_scale, lab_scaler, metadata):
    if lab_scaler is None:
        applied_mean = np.zeros(N_LAB_FEATURES, dtype=np.float64)
        applied_scale = np.ones(N_LAB_FEATURES, dtype=np.float64)
    else:
        applied_mean = lab_scaler.mean_
        applied_scale = lab_scaler.scale_
    np.savez(
        output_dir / "preprocessing_stats.npz",
        lab_scale=np.asarray(lab_scale),
        lab_columns=np.asarray(metadata["lab_columns"], dtype=object),
        encoded_feature_columns=np.asarray(
            metadata["encoded_feature_columns"], dtype=object
        ),
        applied_mean=np.asarray(applied_mean),
        applied_scale=np.asarray(applied_scale),
        yuan_train_source_indices=np.arange(TRAIN_SIZE, dtype=np.int64),
        yuan_test_source_indices=np.arange(
            TRAIN_SIZE, EXPECTED_YUAN_ROWS, dtype=np.int64
        ),
    )


def graph_runtime():
    if torch is None:
        raise ImportError(
            "Legacy baseline execution requires PyTorch, but torch is not installed"
        )
    try:
        from torch_geometric.data import Data
        from torch_geometric.loader import DataLoader
    except ImportError as error:
        raise ImportError("Legacy baseline execution requires torch-geometric") from error
    return Data, DataLoader


def make_legacy_loaders(training_features, test_features, test_labels, batch_size):
    """Build loaders exactly as baseline_paper.py (no explicit generator)."""
    Data, DataLoader = graph_runtime()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = [
        Data(
            x=torch.from_numpy(row).to(torch.float32).unsqueeze(0),
            y=torch.tensor([0], dtype=torch.long),
        ).to(device)
        for row in training_features
    ]
    test_dataset = [
        Data(
            x=torch.from_numpy(row).to(torch.float32).unsqueeze(0),
            y=torch.tensor([int(label)], dtype=torch.long),
        ).to(device)
        for row, label in zip(test_features, test_labels)
    ]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1)
    return train_loader, test_loader


def get_model_factory(method, n_features, seed):
    """Import dependencies before seeding, as in baseline_paper.py."""
    if method == "if":
        return lambda: IsolationForest(random_state=seed)
    if method == "lof":
        try:
            from pyod.models.lof import LOF
        except ImportError as error:
            raise ImportError("LOF requires pyod") from error
        return LOF
    if method == "ocsvm":
        return lambda: OneClassSVM(kernel="rbf", gamma="auto")
    if method == "deepsvdd":
        try:
            from pyod.models.deep_svdd import DeepSVDD
        except ImportError as error:
            raise ImportError("DeepSVDD requires pyod and its torch extras") from error
        return lambda: DeepSVDD(
            n_features,
            use_ae=True,
            hidden_neurons=[32, 32, 32, 8],
            epochs=900,
            batch_size=100,
            dropout_rate=0.0,
        )
    if method == "abod":
        try:
            from pyod.models.abod import ABOD
        except ImportError as error:
            raise ImportError("ABOD requires pyod") from error
        return lambda: ABOD(contamination=0.5)
    if method == "mo_gaal":
        try:
            from pyod.models.mo_gaal import MO_GAAL
        except ImportError as error:
            raise ImportError("MO_GAAL requires pyod and its deep-learning extras") from error
        return lambda: MO_GAAL(
            k=3,
            stop_epochs=100,
            lr_d=0.0005,
            lr_g=0.00001,
            momentum=0.9,
            contamination=0.5,
        )
    if method == "so_gaal":
        try:
            from pyod.models.so_gaal import SO_GAAL
        except ImportError as error:
            raise ImportError("SO_GAAL requires pyod and its deep-learning extras") from error
        return lambda: SO_GAAL(
            stop_epochs=100,
            lr_d=0.0001,
            lr_g=0.00001,
            momentum=0.9,
            contamination=0.5,
        )
    if method == "auto_encoder":
        try:
            from pyod.models.auto_encoder import AutoEncoder
        except ImportError as error:
            raise ImportError("AutoEncoder requires pyod and its torch extras") from error
        return lambda: AutoEncoder(
            contamination=0.5,
            preprocessing=True,
            lr=0.0001,
            batch_size=100,
            hidden_neuron_list=[32, 32, 32, 8],
            batch_norm=False,
            dropout_rate=0.0,
        )
    if method == "dif":
        try:
            from pyod.models.dif import DIF
        except ImportError as error:
            raise ImportError("DIF requires a pyod release containing pyod.models.dif") from error
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return lambda: DIF(device=device)
    if method == "ecod":
        try:
            from pyod.models.ecod import ECOD
        except ImportError as error:
            raise ImportError("ECOD requires pyod") from error
        return lambda: ECOD(contamination=0.5)
    if method == "slad":
        try:
            from algorithms.slad import SLAD
        except ImportError as error:
            raise ImportError(
                "SLAD requires an importable algorithms.slad module"
            ) from error
        return lambda: SLAD(random_state=seed)
    raise ValueError(f"Unsupported method: {method}")


def method_hyperparameters(method, seed):
    """Serializable record of the constructor hyperparameters."""
    configurations = {
        "if": {"random_state": int(seed)},
        "lof": {},
        "ocsvm": {"kernel": "rbf", "gamma": "auto"},
        "deepsvdd": {
            "n_features": EXPECTED_FEATURE_DIM,
            "use_ae": True,
            "hidden_neurons": [32, 32, 32, 8],
            "epochs": 900,
            "batch_size": 100,
            "dropout_rate": 0.0,
        },
        "abod": {"contamination": 0.5},
        "mo_gaal": {
            "k": 3,
            "stop_epochs": 100,
            "lr_d": 0.0005,
            "lr_g": 0.00001,
            "momentum": 0.9,
            "contamination": 0.5,
        },
        "so_gaal": {
            "stop_epochs": 100,
            "lr_d": 0.0001,
            "lr_g": 0.00001,
            "momentum": 0.9,
            "contamination": 0.5,
        },
        "auto_encoder": {
            "contamination": 0.5,
            "preprocessing": True,
            "lr": 0.0001,
            "batch_size": 100,
            "hidden_neurons": [32, 32, 32, 8],
            "batch_norm": False,
            "dropout_rate": 0.0,
            "declared_but_not_passed_train_epoch": 400,
        },
        "dif": {"device": "cuda if available, otherwise cpu"},
        "ecod": {"contamination": 0.5},
        "slad": {},
    }
    return configurations[method]


def collect_training_arrays(train_loader):
    feature_parts = []
    label_parts = []
    for batch in train_loader:
        feature_parts.append(batch.x.detach().cpu())
        label_parts.append(batch.y.detach().cpu())
    features = torch.cat(feature_parts, dim=0).numpy()
    labels = torch.cat(label_parts, dim=0).numpy().reshape(-1)
    return features, labels


def train_model(model, method, train_loader, training_features):
    """Fit a detector, using the full matrix for classical estimators."""
    fit_start = time.perf_counter()
    if method in FULL_MATRIX_CLASSICAL_METHODS:
        # sklearn/PyOD fit() overwrites the previous model. Fit exactly once on
        # all 285 Yuan rows instead of repeatedly fitting random mini-batches.
        model.fit(np.asarray(training_features, dtype=np.float32))
    elif method in REBUTTAL_ARRAY_METHODS:
        training_features, training_labels = collect_training_arrays(train_loader)
        if method == "dif":
            model.fit(training_features, training_labels)
        else:
            model.fit(training_features)
    else:
        raise ValueError(f"Unsupported method: {method}")
    return time.perf_counter() - fit_start


def scalar_score(value, method):
    if torch is not None and torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    values = np.asarray(value).reshape(-1)
    if values.size != 1:
        raise ValueError(
            f"{method} returned {values.size} scores for a batch of size one"
        )
    score = float(values[0])
    if not np.isfinite(score):
        raise ValueError(f"{method} returned a NaN/Inf score")
    return score


def score_model(model, method, test_loader):
    """Return raw continuous model scores in fixed test-set order."""
    score_start = time.perf_counter()
    scores = []
    labels = []
    for batch in test_loader:
        if method in ("if", "lof", "ocsvm") or method in REBUTTAL_ARRAY_METHODS:
            value = model.decision_function(batch.x.detach().cpu().numpy())
        else:
            raise ValueError(f"Unsupported method: {method}")
        scores.append(scalar_score(value, method))
        labels.append(int(batch.y.detach().cpu().numpy().reshape(-1)[0]))
    return (
        np.asarray(scores, dtype=np.float64),
        np.asarray(labels, dtype=np.int64),
        time.perf_counter() - score_start,
    )


def anomaly_scores_from_raw(method, raw_scores):
    """Orient corrected classical scores so larger means more anomalous."""
    raw_scores = np.asarray(raw_scores, dtype=np.float64)
    if method in ("if", "ocsvm"):
        # sklearn IsolationForest and OneClassSVM decision_function values are
        # larger for inliers, so negate them.
        return -raw_scores
    if method == "lof":
        # PyOD LOF follows the PyOD convention: larger means more anomalous.
        return raw_scores
    # Preserve the source orientation for methods outside this targeted fix.
    return -raw_scores


def evaluate(labels, scores):
    auc_value = float(roc_auc_score(labels, scores))
    ap_value = float(average_precision_score(labels, scores))
    fpr, tpr, thresholds = roc_curve(labels, scores)
    distances = np.sqrt((1.0 - tpr) ** 2 + fpr**2)
    best_threshold = float(thresholds[int(np.argmin(distances))])
    return {
        "auc": auc_value,
        "ap": ap_value,
        "best_threshold": best_threshold,
        "rows": int(len(labels)),
        "normal_rows": int(np.sum(labels == 0)),
        "anomaly_rows": int(np.sum(labels == 1)),
    }


def save_roc_plot(labels, scores, title, output_path):
    import matplotlib.pyplot as plt

    fpr, tpr, _ = roc_curve(labels, scores)
    auc_value = roc_auc_score(labels, scores)
    figure, axis = plt.subplots(figsize=(6, 5))
    axis.plot(fpr, tpr, label=f"ROC curve (area={auc_value:.2f})")
    axis.plot([0, 1], [0, 1], "k--")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.05)
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.set_title(title)
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def build_manifest(metadata):
    return pd.concat(
        [
            pd.DataFrame(
                {
                    "test_group": "yuan_heldout",
                    "dynasty": "Yuan",
                    "source_index": np.arange(TRAIN_SIZE, EXPECTED_YUAN_ROWS),
                    "sample_id": metadata["yuan_sample_ids"][TRAIN_SIZE:],
                    "label": 0,
                }
            ),
            pd.DataFrame(
                {
                    "test_group": "ming",
                    "dynasty": "Ming",
                    "source_index": np.arange(EXPECTED_MING_ROWS),
                    "sample_id": metadata["ming_sample_ids"],
                    "label": 1,
                }
            ),
        ],
        ignore_index=True,
    )


def main():
    args = parse_args()
    if args.batch_size <= 0 or args.classical_fit_epochs <= 0:
        raise ValueError("batch-size and classical-fit-epochs must be positive")
    if len(set(args.methods)) != len(args.methods):
        raise ValueError("methods must not contain duplicates")

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (
            ROOT
            / "baseline_paper_corrected_labscale"
            / args.lab_scale
            / f"seed_{args.seed}"
        )
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data_file = args.data_file.resolve()
    yuan_features, ming_features, lab_scaler, metadata = load_legacy_features(
        data_file, args.lab_scale
    )
    training_features = yuan_features[:TRAIN_SIZE]
    test_features = np.concatenate(
        [yuan_features[TRAIN_SIZE:], ming_features], axis=0
    ).astype(np.float32)
    expected_labels = np.concatenate(
        [np.zeros(YUAN_TEST_SIZE), np.ones(EXPECTED_MING_ROWS)]
    ).astype(np.int64)
    if training_features.shape != (TRAIN_SIZE, EXPECTED_FEATURE_DIM):
        raise AssertionError(f"Unexpected training shape: {training_features.shape}")
    if test_features.shape != (50, EXPECTED_FEATURE_DIM):
        raise AssertionError(f"Unexpected test shape: {test_features.shape}")

    save_preprocessing(output_dir, args.lab_scale, lab_scaler, metadata)
    protocol = {
        "source": "baseline_paper.py",
        "comparison_design": "raw and standardized differ only in six LAB columns",
        "data_file": str(data_file),
        "lab_scale": args.lab_scale,
        "lab_columns": [str(column) for column in metadata["lab_columns"]],
        "lab_transform": (
            "StandardScaler fitted on first 285 Yuan rows (ddof=0)"
            if args.lab_scale == "standardized"
            else "identity; source raw LAB values"
        ),
        "categorical_protocol": (
            "legacy source behavior: encode full Yuan and full Ming separately "
            "before dropping the secondary header row"
        ),
        "physical_split": (
            "fixed source order: first 285 Yuan train / last 25 Yuan + 25 Ming test"
        ),
        "classical_fit_protocol": (
            "IF/LOF/OCSVM fit exactly once on the complete 285-row Yuan "
            "training matrix"
        ),
        "deprecated_classical_fit_epochs_argument": args.classical_fit_epochs,
        "rebuttal_method_fit_protocol": (
            "DeepSVDD/ABOD/MO_GAAL/SO_GAAL/AutoEncoder/DIF/ECOD/SLAD: "
            "collect one shuffled full training matrix and fit once"
        ),
        "score_protocol": (
            "continuous decision_function scores; IF and OCSVM are negated "
            "because sklearn assigns larger values to inliers; PyOD LOF is "
            "kept unchanged because larger values indicate anomalies"
        ),
        "seed_protocol": (
            "Python, NumPy, PyTorch, and CUDA seeded; deterministic cuDNN; "
            "IsolationForest random_state set explicitly"
        ),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "methods": args.methods,
        "model_hyperparameters": {
            method: method_hyperparameters(method, args.seed)
            for method in args.methods
        },
    }
    write_json(output_dir / "protocol.json", protocol)

    manifest = build_manifest(metadata)
    results = []
    for method in args.methods:
        model_factory = get_model_factory(
            method, training_features.shape[1], args.seed
        )
        set_global_seed(args.seed)
        train_loader, test_loader = make_legacy_loaders(
            training_features,
            test_features,
            expected_labels,
            args.batch_size,
        )
        model = model_factory()
        fit_seconds = train_model(
            model,
            method,
            train_loader,
            training_features,
        )
        raw_scores, labels, score_seconds = score_model(
            model, method, test_loader
        )
        if not np.array_equal(labels, expected_labels):
            raise AssertionError("Test labels or ordering differ from source protocol")
        anomaly_scores = anomaly_scores_from_raw(method, raw_scores)
        metrics = evaluate(labels, -anomaly_scores)

        predictions = manifest.copy()
        predictions["raw_model_score"] = raw_scores
        predictions["anomaly_score"] = anomaly_scores
        prediction_path = output_dir / f"{method}_test_predictions.csv"
        predictions.to_csv(prediction_path, index=False, encoding="utf-8-sig")
        if not args.no_roc_plot:
            save_roc_plot(
                labels,
                anomaly_scores,
                f"{method}: {args.lab_scale} LAB (corrected protocol)",
                output_dir / f"{method}_roc.png",
            )

        result = {
            "method": method,
            "model_hyperparameters": method_hyperparameters(method, args.seed),
            "lab_scale": args.lab_scale,
            "seed": args.seed,
            "fit_seconds": float(fit_seconds),
            "score_seconds": float(score_seconds),
            "average_score_seconds_per_sample": float(
                score_seconds / len(raw_scores)
            ),
            "test": metrics,
            "prediction_file": str(prediction_path),
        }
        results.append(result)
        write_json(output_dir / f"{method}_summary.json", result)
        print(
            f"{method} | LAB={args.lab_scale} | "
            f"AUC={metrics['auc']:.6f}, AP={metrics['ap']:.6f}"
        )

    write_json(
        output_dir / "all_baseline_results.json",
        {"protocol": protocol, "results": results},
    )
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()
