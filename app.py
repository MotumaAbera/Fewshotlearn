import pandas as pd
import streamlit as st
import torch
from torchvision.transforms.functional import to_pil_image

from config import Config, DATASET_LABELS, SUPPORTED_DATASETS
from dataset import create_datasets
from model import SiameseNetwork
from pipeline import prepare_environment, run_experiment
from utils import load_json


st.set_page_config(
    page_title="Siamese Few-Shot Learning",
    page_icon="MN",
    layout="wide",
)


def build_config() -> Config:
    dataset_name = st.session_state.get("dataset_name", "mnist")
    if dataset_name not in SUPPORTED_DATASETS:
        dataset_name = "mnist"
        st.session_state["dataset_name"] = dataset_name
    config = Config(dataset_name=dataset_name)
    config.epochs = st.session_state.get("epochs", config.epochs)
    config.batch_size = st.session_state.get("batch_size", config.batch_size)
    config.learning_rate = st.session_state.get("learning_rate", config.learning_rate)
    config.one_shot_trials = st.session_state.get("one_shot_trials", config.one_shot_trials)
    config.one_shot_way = st.session_state.get("one_shot_way", config.one_shot_way)
    config.train_pairs_per_epoch = st.session_state.get("train_pairs_per_epoch", config.train_pairs_per_epoch)
    config.val_pairs = st.session_state.get("val_pairs", config.val_pairs)
    config.embedding_dim = st.session_state.get("embedding_dim", config.embedding_dim)
    return config


@st.cache_resource(show_spinner=False)
def load_dataset_bundle(
    cache_version: str,
    dataset_name: str,
    data_dir: str,
    batch_size: int,
    train_pairs_per_epoch: int,
    val_pairs: int,
    num_workers: int,
    seed: int,
):
    config = Config(dataset_name=dataset_name)
    config.data_dir = config.project_root / data_dir
    config.batch_size = batch_size
    config.train_pairs_per_epoch = train_pairs_per_epoch
    config.val_pairs = val_pairs
    config.num_workers = num_workers
    config.seed = seed
    return create_datasets(config)


def model_exists(config: Config) -> bool:
    return (config.models_dir / config.model_name).exists()


def results_exist(config: Config) -> bool:
    return (config.results_dir / config.metrics_name).exists()


def load_saved_model(config: Config, device: torch.device):
    if not model_exists(config):
        return None
    model = SiameseNetwork(embedding_dim=config.embedding_dim, input_channels=config.input_channels).to(device)
    state_dict = torch.load(config.models_dir / config.model_name, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def denormalize_image(image_tensor: torch.Tensor, config: Config) -> torch.Tensor:
    mean, std = config.normalization_stats
    mean_tensor = torch.tensor(mean, dtype=image_tensor.dtype, device=image_tensor.device).view(-1, 1, 1)
    std_tensor = torch.tensor(std, dtype=image_tensor.dtype, device=image_tensor.device).view(-1, 1, 1)
    return torch.clamp(image_tensor * std_tensor + mean_tensor, 0.0, 1.0)


def render_image_comparison(config: Config, device: torch.device) -> None:
    st.subheader("Similarity Explorer")
    try:
        bundle = load_dataset_bundle(
            cache_version="bundle_v3",
            dataset_name=config.dataset_name,
            data_dir=config.data_dir.name,
            batch_size=config.batch_size,
            train_pairs_per_epoch=config.train_pairs_per_epoch,
            val_pairs=config.val_pairs,
            num_workers=config.num_workers,
            seed=config.seed,
        )
    except FileNotFoundError as error:
        st.warning(str(error))
        if config.dataset_name == "ethiopian_textile_local":
            st.code(
                "data/ethiopian_textile_local/\n"
                "|-- train/\n"
                "|   |-- class_a/\n"
                "|   `-- class_b/\n"
                "`-- test/\n"
                "    |-- class_a/\n"
                "    `-- class_b/",
                language="text",
            )
        return

    max_index = len(bundle.raw_test) - 1
    class_names = getattr(bundle, "class_names", [])

    left_col, right_col = st.columns(2)
    with left_col:
        query_index = st.number_input("Query image index", min_value=0, max_value=max_index, value=0, step=1)
    with right_col:
        support_index = st.number_input("Support image index", min_value=0, max_value=max_index, value=1, step=1)

    query_image, query_label = bundle.raw_test[int(query_index)]
    support_image, support_label = bundle.raw_test[int(support_index)]
    query_name = class_names[int(query_label)] if class_names else str(query_label)
    support_name = class_names[int(support_label)] if class_names else str(support_label)

    model = load_saved_model(config, device)
    distance = None
    if model is not None:
        with torch.no_grad():
            query_embedding = model.forward_once(query_image.unsqueeze(0).to(device))
            support_embedding = model.forward_once(support_image.unsqueeze(0).to(device))
            distance = torch.nn.functional.pairwise_distance(query_embedding, support_embedding).item()

    image_col_a, image_col_b, stats_col = st.columns([1, 1, 1.1])
    with image_col_a:
        st.image(to_pil_image(denormalize_image(query_image, config)), caption=f"Query label: {query_name}", width="stretch")
    with image_col_b:
        st.image(to_pil_image(denormalize_image(support_image, config)), caption=f"Support label: {support_name}", width="stretch")
    with stats_col:
        st.metric("Same class", "Yes" if int(query_label) == int(support_label) else "No")
        st.metric("Dataset", config.dataset_title)
        if distance is None:
            st.info("Train the model once to enable embedding distance inspection.")
        else:
            st.metric("Embedding distance", f"{distance:.4f}")
            st.caption("Smaller distances indicate higher similarity in the learned feature space.")


def render_outputs(config: Config) -> None:
    st.subheader("Generated Assets")
    figures = [
        ("Sample pairs", config.figures_dir / config.sample_pairs_name),
        ("Loss curve", config.figures_dir / config.loss_curve_name),
        ("t-SNE plot", config.figures_dir / config.tsne_name),
        ("One-shot examples", config.figures_dir / config.one_shot_examples_name),
    ]
    for title, path in figures:
        if path.exists():
            st.markdown(f"**{title}**")
            st.image(str(path), width="stretch")


def render_metrics(config: Config) -> None:
    st.subheader("Latest Metrics")
    metrics_path = config.results_dir / config.metrics_name
    results_table_path = config.results_dir / config.results_table_name

    if metrics_path.exists():
        metrics = load_json(metrics_path)
        metric_cols = st.columns(4)
        metric_cols[0].metric("One-shot Accuracy", f"{metrics['one_shot_accuracy']:.4f}")
        metric_cols[1].metric("Validation Pair Accuracy", f"{metrics['final_val_pair_accuracy']:.4f}")
        metric_cols[2].metric("Best Val Loss", f"{metrics['best_val_loss']:.4f}")
        metric_cols[3].metric("Epochs Run", str(metrics["epochs_ran"]))
        st.caption(f"Latest dataset: {metrics['dataset']}")
    else:
        st.info("No saved metrics yet. Run training from the sidebar to generate them.")

    if results_table_path.exists():
        st.dataframe(pd.read_csv(results_table_path), width="stretch", hide_index=True)
    failures_path = config.results_dir / config.failures_name
    if failures_path.exists():
        st.markdown("**One-shot failure analysis**")
        st.dataframe(pd.read_csv(failures_path), width="stretch", hide_index=True)


def run_training(config: Config) -> None:
    device = prepare_environment(config)
    try:
        with st.spinner("Training Siamese network and generating report assets..."):
            results = run_experiment(config=config, device=device)
    except FileNotFoundError as error:
        st.error(str(error))
        return
    st.session_state["latest_metrics"] = results["metrics"]
    st.success("Training complete. The dashboard has been refreshed with the newest results.")


def main():
    defaults = Config()
    st.title("Few-Shot Learning with Siamese Networks")
    st.caption("Interactive frontend for training, evaluating, and exploring Siamese similarity embeddings.")

    with st.sidebar:
        st.header("Experiment Controls")
        if st.session_state.get("dataset_name") not in SUPPORTED_DATASETS:
            st.session_state["dataset_name"] = "mnist"
        st.selectbox(
            "Dataset",
            options=list(SUPPORTED_DATASETS),
            key="dataset_name",
            format_func=lambda x: DATASET_LABELS[x],
        )
        st.number_input("Epochs", min_value=1, max_value=50, key="epochs", value=defaults.epochs)
        st.number_input("Batch size", min_value=16, max_value=256, step=16, key="batch_size", value=defaults.batch_size)
        st.number_input("Learning rate", min_value=0.0001, max_value=0.01, step=0.0001, format="%.4f", key="learning_rate", value=defaults.learning_rate)
        st.number_input("One-shot trials", min_value=10, max_value=1000, step=10, key="one_shot_trials", value=defaults.one_shot_trials)
        st.number_input("N-way setting", min_value=2, max_value=10, step=1, key="one_shot_way", value=defaults.one_shot_way)
        st.number_input("Train pairs per epoch", min_value=1000, max_value=50000, step=1000, key="train_pairs_per_epoch", value=defaults.train_pairs_per_epoch)
        st.number_input("Validation pairs", min_value=500, max_value=10000, step=500, key="val_pairs", value=defaults.val_pairs)
        st.number_input("Embedding dimension", min_value=8, max_value=128, step=8, key="embedding_dim", value=defaults.embedding_dim)

        config = build_config()
        if st.button("Run Training", width="stretch"):
            run_training(config)

        st.markdown("**Launch from terminal**")
        st.code("streamlit run app.py", language="bash")

    device = prepare_environment(config)

    overview_col, note_col = st.columns([1.2, 1])
    with overview_col:
        st.markdown(
            """
            This dashboard turns the coursework project into a usable experiment surface.
            Train a Siamese network, inspect saved metrics, and compare individual test images
            through the learned embedding space. You can now switch between MNIST, Fashion-MNIST,
            fashion product images, Amharic handwriting, and local Ethiopian textile datasets,
            inspect one-shot failures, and use the generated report assets directly in your write-up.
            """
        )
    with note_col:
        if model_exists(config):
            st.success("A trained checkpoint is available.")
        else:
            st.warning("No trained checkpoint found yet.")

    tab_summary, tab_explorer, tab_assets = st.tabs(["Results", "Similarity Explorer", "Figures"])
    with tab_summary:
        render_metrics(config)
    with tab_explorer:
        render_image_comparison(config, device)
    with tab_assets:
        render_outputs(config)


if __name__ == "__main__":
    main()
