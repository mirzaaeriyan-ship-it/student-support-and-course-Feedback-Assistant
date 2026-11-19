import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

from train_models import resolve_project_dir, run_training

st.set_page_config(
    page_title="Project 01 — ML & Vision Studio",
    page_icon="🎓",
    layout="wide",
)


@st.cache_data
def load_project_assets(proj_dir_str: str):
    proj_dir = Path(proj_dir_str)
    data_dir = proj_dir / "data"

    meta_path = data_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

    main = pd.read_csv(data_dir / "main.csv")
    target_col = meta.get("target", "needs_support")
    if target_col not in main.columns and "G3" in main.columns:
        main[target_col] = (main["G3"] < 10).astype(int)

    optional_csvs = {}
    for fname in ["feedback.csv", "example_inputs.csv", "image_manifest.csv"]:
        fpath = data_dir / fname
        if fpath.exists():
            optional_csvs[fname] = pd.read_csv(fpath)

    optional_jsons = {}
    for fname in ["prompt_test_cases.json", "topic_keywords.json"]:
        fpath = data_dir / fname
        if fpath.exists():
            optional_jsons[fname] = json.loads(fpath.read_text(encoding="utf-8"))

    images_path = data_dir / "images.npz"
    images_data = None
    if images_path.exists():
        npz = np.load(images_path, allow_pickle=False)
        images_data = {k: npz[k] for k in npz.files}

    return {
        "meta": meta,
        "main": main,
        "target_col": target_col,
        "csvs": optional_csvs,
        "jsons": optional_jsons,
        "images": images_data,
    }


@st.cache_resource
def load_trained_models(proj_dir_str: str):
    models_dir = Path(proj_dir_str) / "models"
    supervised_path = models_dir / "supervised.joblib"
    clusters_path = models_dir / "clusters.joblib"
    cluster_groups_path = models_dir / "cluster_groups.json"
    keras_path = models_dir / "images.keras"

    sup_model = joblib.load(supervised_path) if supervised_path.exists() else None
    clu_model = joblib.load(clusters_path) if clusters_path.exists() else None
    clu_groups = (
        json.loads(cluster_groups_path.read_text(encoding="utf-8"))
        if cluster_groups_path.exists()
        else None
    )
    img_model = tf.keras.models.load_model(keras_path) if keras_path.exists() else None

    return sup_model, clu_model, clu_groups, img_model


def prepare_28x28_grayscale(pil_img: Image.Image, invert: bool = False) -> np.ndarray:
    gray = pil_img.convert("L")
    if invert:
        gray = ImageOps.invert(gray)
    resized = gray.resize((28, 28), Image.Resampling.BILINEAR)
    arr = np.asarray(resized, dtype=np.uint8).reshape(1, 28, 28, 1)
    return arr


# --- Sidebar ---
st.sidebar.title("Project 01 Controls")
try:
    project_dir = resolve_project_dir()
    st.sidebar.success(f"Loaded: `{project_dir.name}`")
except FileNotFoundError as exc:
    st.sidebar.error(str(exc))
    st.stop()

if st.sidebar.button("Train / Refresh All Models & Charts", use_container_width=True):
    with st.spinner("Training Decision Tree, KMeans, and CNN..."):
        run_training(project_dir)
    st.cache_data.clear()
    st.cache_resource.clear()
    st.sidebar.success("Models and reports updated!")

assets = load_project_assets(str(project_dir))
sup_model, clu_model, clu_groups, img_model = load_trained_models(str(project_dir))

meta = assets["meta"]
main_df = assets["main"]
target_col = assets["target_col"]

numeric_features = meta.get("numeric_features", ["G1", "G2", "studytime", "failures"])
categorical_features = meta.get("categorical_features", ["schoolsup", "famsup", "internet"])
cluster_features = meta.get("cluster_features", ["G1", "G2", "studytime", "failures"])
all_features = numeric_features + categorical_features

st.title("🎓 Project 01 — Student Support, Clustering & Digit Classifier")
st.caption("Interactive Streamlit dashboard built from the Project 01 Colab workflow.")

tab_eda, tab_sup, tab_clu, tab_img, tab_assets = st.tabs([
    "1. Data Exploration",
    "2. Supervised Classifier",
    "3. KMeans Clustering",
    "4. CNN Digit Vision",
    "5. Additional Data & JSON",
])

# =====================================================================
# TAB 1: EDA
# =====================================================================
with tab_eda:
    st.header("Dataset Overview & Exploratory Analysis")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Rows", len(main_df))
    c2.metric("Train Rows", int((main_df["split"] == "train").sum()))
    c3.metric("Validation Rows", int((main_df["split"] == "validation").sum()))
    c4.metric("Test Rows", int((main_df["split"] == "test").sum()))

    split_filter = st.selectbox("Filter preview by split", ["all", "train", "validation", "test"])
    preview_df = main_df if split_filter == "all" else main_df[main_df["split"] == split_filter]
    st.dataframe(preview_df.head(25), use_container_width=True)

    with st.expander("Column Types & Missing Values"):
        info_df = pd.DataFrame({
            "dtype": main_df.dtypes.astype(str),
            "missing_count": main_df.isnull().sum(),
            "missing_pct": (main_df.isnull().mean() * 100).round(2),
        })
        st.dataframe(info_df, use_container_width=True)

    train_df = main_df[main_df["split"] == "train"].copy()
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        fig, ax = plt.subplots(figsize=(4.5, 3.2))
        counts = train_df[target_col].value_counts().sort_index()
        ax.bar([str(k) for k in counts.index], counts.values, color=["#4C78A8", "#F58518"])
        ax.set_title(f"Train {target_col}")
        ax.set_ylabel("Students")
        st.pyplot(fig)
        plt.close(fig)

    with col_b:
        fig, ax = plt.subplots(figsize=(4.5, 3.2))
        ax.hist(train_df["G2"], bins=range(0, 22), align="left", rwidth=0.85, color="#54A24B")
        ax.set_title("Train G2 Distribution")
        ax.set_xlabel("G2 (0–20)")
        ax.set_ylabel("Students")
        st.pyplot(fig)
        plt.close(fig)

    with col_c:
        fig, ax = plt.subplots(figsize=(4.5, 3.2))
        g2_by_study = train_df.groupby("studytime")["G2"].mean().sort_index()
        ax.bar([str(k) for k in g2_by_study.index], g2_by_study.values, color="#E45756")
        ax.set_title("Mean G2 by studytime")
        ax.set_xlabel("studytime (1–4)")
        ax.set_ylabel("Mean G2")
        st.pyplot(fig)
        plt.close(fig)

# =====================================================================
# TAB 2: SUPERVISED MODEL
# =====================================================================
with tab_sup:
    st.header("Supervised Support Flag Prediction (DecisionTreeClassifier)")

    report_path = project_dir / "reports" / "supervised_test_results.json"
    if report_path.exists():
        sup_report = json.loads(report_path.read_text(encoding="utf-8"))
        st.subheader("Saved Test Set Benchmarks")
        m1, m2, m3, m4 = st.columns(4)
        dt_m = sup_report["decision_tree_test"]
        bl_m = sup_report["baseline_test"]
        m1.metric("Decision Tree Accuracy", f"{dt_m['accuracy']:.4f}", f"{dt_m['accuracy'] - bl_m['accuracy']:+.4f} vs Dummy")
        m2.metric("Decision Tree Macro F1", f"{dt_m['macro_f1']:.4f}", f"{dt_m['macro_f1'] - bl_m['macro_f1']:+.4f} vs Dummy")
        m3.metric("Class 1 Precision", f"{dt_m['class_1_precision']:.4f}")
        m4.metric("Class 1 Recall", f"{dt_m['class_1_recall']:.4f}")

        if sup_report.get("sample_test_errors_up_to_5"):
            with st.expander("Sample Test Misclassifications (up to 5)"):
                st.dataframe(pd.DataFrame(sup_report["sample_test_errors_up_to_5"]), use_container_width=True)

    st.subheader("Interactive Student Prediction")
    if sup_model is None:
        st.warning("`models/supervised.joblib` not found. Click **Train / Refresh All Models & Charts** in the sidebar.")
    else:
        input_vals = {}
        left_col, right_col = st.columns(2)
        with left_col:
            for col in numeric_features:
                col_series = main_df[col]
                min_v = int(col_series.min())
                max_v = int(col_series.max())
                default_v = int(col_series.median())
                input_vals[col] = st.slider(col, min_value=min_v, max_value=max_v, value=default_v)

        with right_col:
            for col in categorical_features:
                options = sorted(main_df[col].dropna().astype(str).unique().tolist())
                input_vals[col] = st.selectbox(col, options=options)

        if "example_inputs.csv" in assets["csvs"]:
            ex_df = assets["csvs"]["example_inputs.csv"]
            if st.button("Load First Row from example_inputs.csv"):
                row0 = ex_df.iloc[0].to_dict()
                for k in all_features:
                    if k in row0:
                        input_vals[k] = row0[k]

        input_df = pd.DataFrame([input_vals])[all_features]
        pred_class = int(sup_model.predict(input_df)[0])
        pred_proba = (
            float(sup_model.predict_proba(input_df)[0][1])
            if hasattr(sup_model, "predict_proba")
            else None
        )

        if pred_class == 1:
            st.error(
                f"Prediction: **{target_col} = 1 (Support Flagged)**"
                + (f" — Probability: {pred_proba:.2%}" if pred_proba is not None else "")
            )
        else:
            st.success(
                f"Prediction: **{target_col} = 0 (No Support Flag)**"
                + (f" — Support Probability: {pred_proba:.2%}" if pred_proba is not None else "")
            )

# =====================================================================
# TAB 3: KMEANS CLUSTERING
# =====================================================================
with tab_clu:
    st.header("Student Profile Clustering (KMeans, k=3)")

    if clu_model is None or clu_groups is None:
        st.warning("Cluster artifacts not found. Click **Train / Refresh All Models & Charts** in the sidebar.")
    else:
        st.subheader("Cluster Summaries")
        summary_rows = []
        for gid, info in clu_groups.items():
            row = {"group": gid, "size": info["size"], "description": info["description"]}
            row.update(info["means"])
            summary_rows.append(row)
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

        train_df = main_df[main_df["split"] == "train"].copy()
        train_df["group"] = clu_model.predict(train_df[cluster_features])

        fig, ax = plt.subplots(figsize=(6.5, 4))
        for g in sorted(train_df["group"].unique()):
            sub = train_df[train_df["group"] == g]
            ax.scatter(sub["G2"], sub["studytime"], label=f"Group {g} (n={len(sub)})", alpha=0.75)
        ax.set_title("KMeans Groups (k=3) on Training Set")
        ax.set_xlabel("G2 score (0 to 20 points)")
        ax.set_ylabel("studytime (1 to 4)")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Assign a Custom Student to a Cluster")
        c_inputs = {}
        cols = st.columns(len(cluster_features))
        for idx, col in enumerate(cluster_features):
            s = main_df[col]
            c_inputs[col] = cols[idx].number_input(
                col,
                min_value=float(s.min()),
                max_value=float(s.max()),
                value=float(round(s.mean(), 1)),
                step=0.5,
            )
        assigned_group = int(clu_model.predict(pd.DataFrame([c_inputs])[cluster_features])[0])
        desc = clu_groups.get(str(assigned_group), {}).get("description", f"Group {assigned_group}")
        st.info(f"Assigned to **Group {assigned_group}** — {desc}")

# =====================================================================
# TAB 4: CNN VISION
# =====================================================================
with tab_img:
    st.header("28×28 Grayscale Digit Classifier (CNN)")

    if img_model is None:
        st.warning("`models/images.keras` not found. Click **Train / Refresh All Models & Charts** in the sidebar.")
    else:
        img_report_path = project_dir / "reports" / "image_test_results.json"
        if img_report_path.exists():
            ir = json.loads(img_report_path.read_text(encoding="utf-8"))
            ic1, ic2 = st.columns(2)
            ic1.metric("Test Accuracy", f"{ir['test_accuracy']:.4f}")
            ic2.metric("Test Macro F1", f"{ir['test_macro_f1']:.4f}")

        sub_test, sub_upload = st.columns(2)

        with sub_test:
            st.subheader("Test Set Sample")
            images_dict = assets["images"]
            if images_dict is not None and "x_test" in images_dict:
                x_test = images_dict["x_test"]
                y_test = images_dict["y_test"]
                sample_idx = st.slider("Select x_test index", 0, len(x_test) - 1, 0)
                sample_arr = x_test[sample_idx]
                probs = img_model.predict(sample_arr.reshape(1, 28, 28, 1), verbose=0)[0]
                pred_digit = int(np.argmax(probs))

                fig, ax = plt.subplots(figsize=(3, 3))
                ax.imshow(sample_arr.squeeze(), cmap="gray", vmin=0, vmax=255)
                ax.set_title(f"True: {y_test[sample_idx]} | Pred: {pred_digit}")
                ax.axis("off")
                st.pyplot(fig)
                plt.close(fig)

                st.bar_chart(pd.DataFrame({"probability": probs}, index=[str(i) for i in range(10)]))

        with sub_upload:
            st.subheader("Upload Your Own Digit Image")
            uploaded_file = st.file_uploader("Upload PNG/JPG", type=["png", "jpg", "jpeg"])
            invert_colors = st.checkbox("Invert colors (use if dark digit on light background)", value=False)
            if uploaded_file is not None:
                pil_img = Image.open(uploaded_file)
                arr_28 = prepare_28x28_grayscale(pil_img, invert=invert_colors)
                probs = img_model.predict(arr_28, verbose=0)[0]
                pred_digit = int(np.argmax(probs))

                st.image(arr_28.reshape(28, 28), caption=f"28×28 Input — Predicted: {pred_digit} ({probs[pred_digit]:.2%})", width=160)
                st.bar_chart(pd.DataFrame({"probability": probs}, index=[str(i) for i in range(10)]))

# =====================================================================
# TAB 5: ADDITIONAL DATA & JSON
# =====================================================================
with tab_assets:
    st.header("Supporting CSVs, JSON Configs & Sample Images")

    for cname, cdf in assets["csvs"].items():
        if cname != "example_inputs.csv":
            with st.expander(f"CSV: {cname} ({len(cdf)} rows)"):
                st.dataframe(cdf, use_container_width=True)

    for jname, jdata in assets["jsons"].items():
        with st.expander(f"JSON: {jname}"):
            st.json(jdata)

    sample_imgs_dir = project_dir / "sample_images"
    if sample_imgs_dir.exists():
        pngs = sorted(sample_imgs_dir.glob("*.png"))
        if pngs:
            with st.expander(f"Sample Images ({len(pngs)} files)"):
                st.image([str(p) for p in pngs], caption=[p.name for p in pngs], width=110)