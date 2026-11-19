import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.compose import make_column_transformer
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


def resolve_project_dir(base_dir: Path | None = None) -> Path:
    root = base_dir or Path(__file__).resolve().parent
    candidates = [
        root / "Project_01",
        root / "extracted_pack" / "Project_01",
        root,
    ]
    for cand in candidates:
        if (cand / "data" / "main.csv").exists():
            return cand
    raise FileNotFoundError("Could not locate Project_01/data/main.csv.")


def run_training(project_dir: Path | None = None) -> dict:
    proj_dir = resolve_project_dir(project_dir)
    data_dir = proj_dir / "data"
    models_dir = proj_dir / "models"
    reports_dir = proj_dir / "reports"
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    main = pd.read_csv(data_dir / "main.csv")
    images = np.load(data_dir / "images.npz", allow_pickle=False)

    meta_path = data_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

    numeric_features = meta.get("numeric_features", ["G1", "G2", "studytime", "failures"])
    categorical_features = meta.get("categorical_features", ["schoolsup", "famsup", "internet"])
    cluster_features = meta.get("cluster_features", ["G1", "G2", "studytime", "failures"])
    target_col = meta.get("target", "needs_support")

    if target_col not in main.columns and "G3" in main.columns:
        main[target_col] = (main["G3"] < 10).astype(int)

    features = numeric_features + categorical_features

    train = main[main["split"] == "train"].copy()
    val = main[main["split"] == "validation"].copy()
    test = main[main["split"] == "test"].copy()

    # 1. Charts
    fig, ax = plt.subplots(figsize=(5, 3.5))
    counts = train[target_col].value_counts().sort_index()
    ax.bar([str(k) for k in counts.index], counts.values, color=["#4C78A8", "#F58518"])
    ax.set_title("Training Set Support Flag Counts")
    ax.set_xlabel(f"{target_col} (0 = no support flag, 1 = support flag)")
    ax.set_ylabel("Number of students")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 2, str(v), ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(reports_dir / "chart1_target_distribution.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.hist(train["G2"], bins=range(0, 22), align="left", rwidth=0.85, color="#54A24B")
    ax.set_title("Training Set Second-Period Mark (G2) Distribution")
    ax.set_xlabel("G2 score (0 to 20 points)")
    ax.set_ylabel("Number of students")
    plt.tight_layout()
    plt.savefig(reports_dir / "chart2_g2_histogram.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    g2_by_study = train.groupby("studytime")["G2"].mean().sort_index()
    ax.bar([str(k) for k in g2_by_study.index], g2_by_study.values, color="#E45756")
    ax.set_title("Average G2 by Reported Study Time (Training Set)")
    ax.set_xlabel("studytime category (1=low to 4=high)")
    ax.set_ylabel("Mean G2 score (points)")
    for i, v in enumerate(g2_by_study.values):
        ax.text(i, v + 0.25, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(reports_dir / "chart3_g2_by_studytime.png", dpi=150)
    plt.close(fig)

    # 2. Supervised Baseline & Decision Tree
    encoder = make_column_transformer(
        (OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        remainder="passthrough",
    )
    baseline = make_pipeline(
        encoder,
        DummyClassifier(strategy="most_frequent", random_state=42),
    )
    baseline.fit(train[features], train[target_col])

    supervised_model = make_pipeline(
        make_column_transformer(
            (OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
            remainder="passthrough",
        ),
        DecisionTreeClassifier(max_depth=4, random_state=42),
    )
    supervised_model.fit(train[features], train[target_col])

    val_pred = supervised_model.predict(val[features])
    val_metrics = {
        "val_accuracy": float(accuracy_score(val[target_col], val_pred)),
        "val_macro_f1": float(f1_score(val[target_col], val_pred, average="macro", zero_division=0)),
    }

    def eval_classifier(model, X_eval, y_eval):
        preds = model.predict(X_eval)
        acc = accuracy_score(y_eval, preds)
        mf1 = f1_score(y_eval, preds, average="macro", zero_division=0)
        p1 = precision_score(y_eval, preds, pos_label=1, zero_division=0)
        r1 = recall_score(y_eval, preds, pos_label=1, zero_division=0)
        cm = confusion_matrix(y_eval, preds, labels=[0, 1])
        return preds, {
            "accuracy": float(acc),
            "macro_f1": float(mf1),
            "class_1_precision": float(p1),
            "class_1_recall": float(r1),
            "confusion_matrix_0_1": cm.tolist(),
        }

    _, base_metrics = eval_classifier(baseline, test[features], test[target_col])
    test_preds, tree_metrics = eval_classifier(supervised_model, test[features], test[target_col])

    joblib.dump(supervised_model, models_dir / "supervised.joblib")

    errors_df = test.copy()
    errors_df["pred"] = test_preds
    errors_df = errors_df[errors_df[target_col] != errors_df["pred"]]
    sample_errors = errors_df.head(5).to_dict(orient="records")

    supervised_report = {
        "features": features,
        "target": target_col,
        "validation": val_metrics,
        "baseline_test": base_metrics,
        "decision_tree_test": tree_metrics,
        "sample_test_errors_up_to_5": sample_errors,
    }
    (reports_dir / "supervised_test_results.json").write_text(
        json.dumps(supervised_report, indent=2, default=str),
        encoding="utf-8",
    )

    # 3. KMeans Clustering
    cluster_model = make_pipeline(
        StandardScaler(),
        KMeans(n_clusters=3, n_init=10, random_state=42),
    )
    group_ids = cluster_model.fit_predict(train[cluster_features])

    profiles = train[cluster_features].copy()
    profiles["group"] = group_ids
    group_sizes = profiles.groupby("group").size()
    group_means = profiles.groupby("group")[cluster_features].mean()

    fig, ax = plt.subplots(figsize=(6, 4))
    for g in sorted(profiles["group"].unique()):
        sub = profiles[profiles["group"] == g]
        ax.scatter(sub["G2"], sub["studytime"], label=f"Group {g} (n={len(sub)})", alpha=0.75)
    ax.set_title("KMeans Groups (k=3) on Training Set")
    ax.set_xlabel("G2 score (0 to 20 points)")
    ax.set_ylabel("studytime (1 to 4)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(reports_dir / "chart4_kmeans_scatter.png", dpi=150)
    plt.close(fig)

    joblib.dump(cluster_model, models_dir / "clusters.joblib")

    group_descriptions = {}
    for g in sorted(profiles["group"].unique()):
        row = group_means.loc[g]
        group_descriptions[str(g)] = {
            "size": int(group_sizes.loc[g]),
            "means": {col: round(float(row[col]), 2) for col in cluster_features},
            "description": (
                f"Group {g} (n={int(group_sizes.loc[g])}): mean G1={row['G1']:.1f}, "
                f"G2={row['G2']:.1f}, studytime={row['studytime']:.1f}, failures={row['failures']:.2f}."
            ),
        }

    (models_dir / "cluster_groups.json").write_text(
        json.dumps(group_descriptions, indent=2),
        encoding="utf-8",
    )

    # 4. CNN Image Classifier
    tf.keras.utils.set_random_seed(42)

    fig, axes = plt.subplots(1, 5, figsize=(10, 2.2))
    for i in range(5):
        axes[i].imshow(images["x_train"][i].squeeze(), cmap="gray", vmin=0, vmax=255)
        axes[i].set_title(f"Label: {images['y_train'][i]}")
        axes[i].axis("off")
    plt.tight_layout()
    plt.savefig(reports_dir / "chart5_sample_digits.png", dpi=150)
    plt.close(fig)

    image_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Rescaling(1.0 / 255),
        tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])

    image_model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=1, restore_best_weights=True
    )

    history = image_model.fit(
        images["x_train"],
        images["y_train"],
        validation_data=(images["x_validation"], images["y_validation"]),
        epochs=3,
        batch_size=32,
        callbacks=[stop],
        verbose=0,
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))
    epochs_ran = range(1, len(history.history["loss"]) + 1)
    ax1.plot(epochs_ran, history.history["accuracy"], marker="o", label="train_acc")
    ax1.plot(epochs_ran, history.history["val_accuracy"], marker="o", label="val_acc")
    ax1.set_title("CNN Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()

    ax2.plot(epochs_ran, history.history["loss"], marker="o", label="train_loss")
    ax2.plot(epochs_ran, history.history["val_loss"], marker="o", label="val_loss")
    ax2.set_title("CNN Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    plt.tight_layout()
    plt.savefig(reports_dir / "chart6_cnn_curves.png", dpi=150)
    plt.close(fig)

    test_probs = image_model.predict(images["x_test"], verbose=0)
    test_img_preds = np.argmax(test_probs, axis=1)
    y_img_test = images["y_test"]

    img_acc = float(accuracy_score(y_img_test, test_img_preds))
    img_mf1 = float(f1_score(y_img_test, test_img_preds, average="macro", zero_division=0))
    img_cm = confusion_matrix(y_img_test, test_img_preds, labels=list(range(10))).tolist()

    image_model.save(models_dir / "images.keras")

    img_report = {
        "test_accuracy": img_acc,
        "test_macro_f1": img_mf1,
        "confusion_matrix_0_9": img_cm,
    }
    (reports_dir / "image_test_results.json").write_text(
        json.dumps(img_report, indent=2),
        encoding="utf-8",
    )

    return {
        "supervised": supervised_report,
        "clusters": group_descriptions,
        "images": img_report,
    }


if __name__ == "__main__":
    summary = run_training()
    print("Training complete. Summary keys:", list(summary.keys()))