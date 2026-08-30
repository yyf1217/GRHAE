import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from GWHAE_hnn import HNN_gram


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_FILE = ROOT / "data" / "青花瓷色度统计LAB.xlsx"

SHEET_NAMES = ["元代", "明代"]
SAMPLE_COL = "样品号"
VESSEL_COL = "器型"
COLOR_COL = "青花呈色"

EXPECTED_YUAN_ROWS = 310
EXPECTED_MING_ROWS = 25
TRAIN_SIZE = 285
YUAN_TEST_SIZE = 25
N_LAB_FEATURES = 6
EXPECTED_FEATURE_DIM = 32
SPLIT_SEED = 12393


MODEL_CONFIG = {
    "n_hidden": 16,
    "n_layers": 2,
    "out_features": 8,
    "dropout": 0.1,
    "lr": 0.00008,
    "c": 0.019,
    "beta": 6,    # 9.021041917714655
}

LAB_SCALE = "standardized"
DEFAULT_OUTPUT_DIR = ROOT / "train_model" / "hnn_gram_git_standardized"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", type=Path, default=DEFAULT_DATA_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1236)
    parser.add_argument("--verbose-training", action="store_true")
    return parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def make_one_hot_encoder():
    """Dense OneHotEncoder compatible with older and newer sklearn."""
    try:
        return OneHotEncoder(
            handle_unknown="ignore", sparse_output=False, dtype=np.float32
        )
    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore", sparse=False, dtype=np.float32
        )


def load_raw_data(data_file):
    worksheets = pd.read_excel(data_file, sheet_name=SHEET_NAMES)

    yuan = worksheets["元代"].iloc[1:].copy().reset_index(drop=True)
    ming = worksheets["明代"].iloc[1:].copy().reset_index(drop=True)
    if len(yuan) != EXPECTED_YUAN_ROWS or len(ming) != EXPECTED_MING_ROWS:
        raise ValueError(
            "Expected 310 Yuan and 25 Ming physical rows, found "
            f"{len(yuan)} and {len(ming)}"
        )

    expected_metadata = [SAMPLE_COL, VESSEL_COL, COLOR_COL]
    if list(yuan.columns[:3]) != expected_metadata:
        raise ValueError(
            f"Unexpected metadata columns: {list(yuan.columns[:3])}; "
            f"expected {expected_metadata}"
        )


    lab_columns = list(yuan.columns[3 : 3 + N_LAB_FEATURES])
    if len(lab_columns) != N_LAB_FEATURES:
        raise ValueError(f"Expected six LAB columns, got {lab_columns}")
    if list(ming.columns[3 : 3 + N_LAB_FEATURES]) != lab_columns:
        raise ValueError("Yuan and Ming worksheets have different LAB columns")

    for name, frame in (("Yuan", yuan), ("Ming", ming)):
        for column in lab_columns:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        if frame[lab_columns].isna().any().any():
            raise ValueError(f"{name} contains missing LAB values")
    return yuan, ming, lab_columns


def fixed_physical_split(yuan):
    permutation = np.random.default_rng(SPLIT_SEED).permutation(len(yuan))
    train_indices = permutation[:TRAIN_SIZE]
    test_indices = permutation[TRAIN_SIZE:]
    if len(train_indices) != TRAIN_SIZE or len(test_indices) != YUAN_TEST_SIZE:
        raise AssertionError("Expected a 285/25 Yuan split")
    if not set(train_indices).isdisjoint(set(test_indices)):
        raise AssertionError("Training and held-out Yuan rows overlap")
    return (
        yuan.iloc[train_indices].reset_index(drop=True),
        yuan.iloc[test_indices].reset_index(drop=True),
        train_indices,
        test_indices,
    )


def fit_training_preprocessors(training_frame, lab_columns, lab_scale):
    color_encoder = LabelEncoder().fit(training_frame[COLOR_COL].astype(str))
    if len(color_encoder.classes_) != 2:
        raise ValueError(
            f"Expected two decoration-colour classes, got {color_encoder.classes_}"
        )

    vessel_encoder = make_one_hot_encoder()
    vessel_encoder.fit(training_frame[[VESSEL_COL]].astype(str))
    if len(vessel_encoder.categories_[0]) != 25:
        raise ValueError(
            "Expected 25 vessel categories in the training pool, found "
            f"{len(vessel_encoder.categories_[0])}"
        )

    lab_scaler = None
    if lab_scale == "standardized":
        lab_scaler = StandardScaler().fit(
            training_frame[lab_columns].to_numpy(dtype=np.float32)
        )
    elif lab_scale != "raw":
        raise ValueError(f"Unknown LAB scale: {lab_scale}")
    return lab_scaler, color_encoder, vessel_encoder


def transform_color(encoder, values):
    values = values.astype(str)
    unseen = sorted(set(values.unique()) - set(encoder.classes_))
    if unseen:
        raise ValueError(f"Unseen decoration colours in held-out data: {unseen}")
    return encoder.transform(values).astype(np.float32).reshape(-1, 1)


def transform_features(
    frame,
    lab_columns,
    lab_scale,
    lab_scaler,
    color_encoder,
    vessel_encoder,
):
    lab = frame[lab_columns].to_numpy(dtype=np.float32)
    if lab_scale == "standardized":
        lab = lab_scaler.transform(lab).astype(np.float32)
    color = transform_color(color_encoder, frame[COLOR_COL])
    vessel = vessel_encoder.transform(
        frame[[VESSEL_COL]].astype(str)
    ).astype(np.float32)
    features = np.concatenate([lab, color, vessel], axis=1).astype(np.float32)
    if features.shape[1] != EXPECTED_FEATURE_DIM:
        raise ValueError(
            f"Expected 32 features (6 LAB + 1 colour + 25 vessel), got "
            f"{features.shape[1]}"
        )
    return features


def make_loader(features, labels, batch_size, shuffle, seed):
    if np.isscalar(labels):
        labels = np.full(len(features), int(labels), dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    if len(features) != len(labels):
        raise ValueError("Feature and label counts differ")
    dataset = [
        Data(
            x=torch.from_numpy(row).unsqueeze(0),
            y=torch.tensor([int(label)], dtype=torch.long),
        )
        for row, label in zip(features, labels)
    ]
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )


def expmap_radius_diagnostics(features, curvature):
    raw_norm = np.linalg.norm(features, axis=1)
    normalized_radius = np.tanh(np.sqrt(curvature) * raw_norm)
    normalized_radius = np.minimum(normalized_radius, 0.996)
    return {
        "input_norm_median": float(np.median(raw_norm)),
        "rho_median": float(np.median(normalized_radius)),
        "rho_p95": float(np.percentile(normalized_radius, 95)),
        "rho_max": float(np.max(normalized_radius)),
        "fraction_rho_gt_0.95": float(np.mean(normalized_radius > 0.95)),
        "fraction_rho_gt_0.99": float(np.mean(normalized_radius > 0.99)),
    }


def evaluate(model, test_loader):
    scores = []
    labels = []
    latents = []
    for batch in test_loader:
        batch_scores, batch_latents = model.decision_function(batch)
        scores.append(np.asarray(batch_scores))
        labels.append(batch.y.detach().cpu().numpy().reshape(-1))
        latents.append(batch_latents.detach().cpu().numpy())
    scores = np.concatenate(scores)
    labels = np.concatenate(labels).astype(np.int64)
    latents = np.concatenate(latents, axis=0)
    return {
        "auc": float(roc_auc_score(labels, scores)),
        "ap": float(average_precision_score(labels, scores)),
        "scores": scores,
        "labels": labels,
        "latents": latents,
    }


def save_preprocessing(
    output_dir,
    lab_scale,
    lab_columns,
    lab_scaler,
    color_encoder,
    vessel_encoder,
    train_indices,
    test_indices,
):
    if lab_scaler is None:
        applied_mean = np.zeros(N_LAB_FEATURES, dtype=np.float64)
        applied_scale = np.ones(N_LAB_FEATURES, dtype=np.float64)
    else:
        applied_mean = lab_scaler.mean_
        applied_scale = lab_scaler.scale_
    np.savez(
        output_dir / "preprocessing_stats.npz",
        lab_scale=np.asarray(lab_scale),
        lab_columns=np.asarray(lab_columns, dtype=object),
        applied_mean=applied_mean,
        applied_scale=applied_scale,
        color_classes=np.asarray(color_encoder.classes_, dtype=object),
        vessel_categories=np.asarray(vessel_encoder.categories_[0], dtype=object),
        yuan_train_source_indices=np.asarray(train_indices, dtype=np.int64),
        yuan_test_source_indices=np.asarray(test_indices, dtype=np.int64),
    )

def main():
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("Epoch count and batch size must be positive")
    set_seed(args.seed)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    yuan, ming, lab_columns = load_raw_data(args.data_file.resolve())
    yuan_train, yuan_test, train_indices, test_indices = fixed_physical_split(yuan)
    lab_scaler, color_encoder, vessel_encoder = fit_training_preprocessors(
        yuan_train, lab_columns, LAB_SCALE
    )

    train_features = transform_features(
        yuan_train,
        lab_columns,
        LAB_SCALE,
        lab_scaler,
        color_encoder,
        vessel_encoder,
    )
    yuan_test_features = transform_features(
        yuan_test,
        lab_columns,
        LAB_SCALE,
        lab_scaler,
        color_encoder,
        vessel_encoder,
    )
    ming_test_features = transform_features(
        ming,
        lab_columns,
        LAB_SCALE,
        lab_scaler,
        color_encoder,
        vessel_encoder,
    )
    test_features = np.concatenate(
        [yuan_test_features, ming_test_features], axis=0
    ).astype(np.float32)
    test_labels = np.concatenate(
        [np.zeros(YUAN_TEST_SIZE), np.ones(EXPECTED_MING_ROWS)]
    ).astype(np.int64)

    expected_train_shape = (TRAIN_SIZE, EXPECTED_FEATURE_DIM)
    expected_test_shape = (
        YUAN_TEST_SIZE + EXPECTED_MING_ROWS,
        EXPECTED_FEATURE_DIM,
    )
    if train_features.shape != expected_train_shape:
        raise AssertionError(
            f"Expected training shape {expected_train_shape}, got "
            f"{train_features.shape}"
        )
    if test_features.shape != expected_test_shape:
        raise AssertionError(
            f"Expected test shape {expected_test_shape}, got {test_features.shape}"
        )

    save_preprocessing(
        output_dir,
        LAB_SCALE,
        lab_columns,
        lab_scaler,
        color_encoder,
        vessel_encoder,
        train_indices,
        test_indices,
    )
    radius_stats = expmap_radius_diagnostics(train_features, MODEL_CONFIG["c"])
    protocol = {
        "entry_point": Path(__file__).name,
        "base_pipeline": "GWHAE_git.py standardized branch",
        "lab_columns": [str(column) for column in lab_columns],
        "lab_scale": LAB_SCALE,
        "lab_transform": "training-only StandardScaler z-score (ddof=0)",
        "lab_fit_scope": "285 Yuan training rows only",
        "categorical_fit_scope": "285 Yuan training rows only",
        "split": "fixed 285 Yuan train / 25 Yuan + 25 Ming held-out test",
        "gm": "Eq.7 with model curvature -c and off-diagonal pairs",
        "inference": "Geoopt k=-c and Eq.10 proj_B(h*a_j)",
        "expmap_radius": radius_stats,
    }
    with (output_dir / "protocol.json").open("w", encoding="utf-8") as stream:
        json.dump(protocol, stream, ensure_ascii=False, indent=2)

    train_loader = make_loader(
        train_features,
        0,
        args.batch_size,
        shuffle=True,
        seed=args.seed,
    )
    test_loader = make_loader(
        test_features,
        test_labels,
        args.batch_size,
        shuffle=False,
        seed=args.seed,
    )
    model = HNN_gram(
        **MODEL_CONFIG,
        epoch=args.epochs,
        gpu=0 if torch.cuda.is_available() else -1,
        verbose=args.verbose_training,
        checkpoint_path=str(output_dir / "model.pth"),
    )

    print(f"LAB scale: {LAB_SCALE} (fixed)")
    print(f"LAB columns: {lab_columns}")
    print(f"Expmap radius: {radius_stats}")
    model.fit(train_loader, train_features.shape[1])

    result = evaluate(model, test_loader)
    test_manifest = pd.concat(
        [
            pd.DataFrame(
                {
                    "dynasty": "Yuan",
                    "source_index": test_indices,
                    "sample_id": yuan_test[SAMPLE_COL].astype(str).to_numpy(),
                }
            ),
            pd.DataFrame(
                {
                    "dynasty": "Ming",
                    "source_index": np.arange(len(ming)),
                    "sample_id": ming[SAMPLE_COL].astype(str).to_numpy(),
                }
            ),
        ],
        ignore_index=True,
    )
    test_manifest["label"] = result["labels"]
    test_manifest["anomaly_score"] = result["scores"]
    test_manifest.to_csv(
        output_dir / "test_predictions.csv", index=False, encoding="utf-8-sig"
    )
    np.savez(
        output_dir / "test_latents.npz",
        latent=result["latents"],
        labels=result["labels"],
        scores=result["scores"],
    )

    summary = {
        "lab_scale": LAB_SCALE,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "model_config": dict(MODEL_CONFIG),
        "train_shape": list(train_features.shape),
        "test_shape": list(test_features.shape),
        "auc": result["auc"],
        "ap": result["ap"],
        "expmap_radius": radius_stats,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)

    print(f"Held-out AUC={result['auc']:.6f}, AP={result['ap']:.6f}")
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()
