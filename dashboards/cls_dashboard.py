from glob import glob

import torch
import numpy as np
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import RobertaConfig, RobertaForMaskedLM

from data.dataset import MidiDataset
from data.quantizer import MidiQuantizer
from data.tokenizer import QuantizedMidiEncoder


def preprocess_dataset(
    dataset_name: str,
    quantizer: MidiQuantizer,
    tokenizer: QuantizedMidiEncoder,
    queries: list[str],
):
    dataset = load_dataset(dataset_name, split="validation")

    ds = MidiDataset(dataset, quantizer, tokenizer, masking_probability=0.0)

    idx = {}

    for query in queries:
        idx_query = {i: query for i, name in enumerate(ds.dataset["source"]) if str.lower(query) in str.lower(name)}
        idx.update(idx_query)

    ds = Subset(ds, indices=list(idx.keys()))

    return ds, idx


def plot_embeddings(X: np.ndarray, labels: list[str]):
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)

    fig = plt.figure(figsize=(10, 10))
    sns.scatterplot(x=X_reduced[:, 0], y=X_reduced[:, 1], hue=labels)
    st.pyplot(fig)


def plot_confusion_matrix(cm: np.ndarray):
    fig = plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True)
    st.pyplot(fig)


def plot_class_balance(y: np.ndarray):
    fig = plt.figure(figsize=(10, 10))
    sns.countplot(x=y)
    st.pyplot(fig)


def evaluate_classification(
    model: RobertaForMaskedLM,
    dataset_name: str,
    quantizer: MidiQuantizer,
    tokenizer: QuantizedMidiEncoder,
    queries: list[str],
    device: torch.device,
):
    ds, idx_to_query = preprocess_dataset(
        dataset_name,
        quantizer=quantizer,
        tokenizer=tokenizer,
        queries=queries,
    )
    query_to_label = {k: i for i, k in enumerate(queries)}

    dataloader = DataLoader(ds, batch_size=1024, num_workers=8, shuffle=False)

    outputs = []

    for batch in dataloader:
        with torch.no_grad():
            input_token_ids = batch["input_token_ids"].to(device)

            # shape [batch_size, seq_len, vocab_size]
            out = model(input_ids=input_token_ids).logits
            cls_embedding = out[:, 0, :]
            outputs.append(cls_embedding)

    # shape: [num_sequences, vocab_size]
    X = torch.cat(outputs, dim=0).cpu().numpy()
    labels = np.array(list(idx_to_query.values()))
    y = np.array([query_to_label[q] for q in labels])

    # assert len(X) == len(y)

    plot_class_balance(labels)
    plot_embeddings(X, labels)

    # reducing dimensions
    pca = PCA(n_components=128)
    X_reduced = pca.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)

    tree = DecisionTreeClassifier()
    tree.fit(X_train, y_train)

    acc = tree.score(X_test, y_test)
    cm = confusion_matrix(tree.predict(X_test), y_test)
    plot_confusion_matrix(cm)

    return acc


def main():
    available_checkpoints = glob("checkpoints/*pt")
    ckpt_path = st.selectbox(label="Available Checkpoints", options=available_checkpoints)

    checkpoint = torch.load(ckpt_path)

    cfg = checkpoint["config"]
    device = cfg.train.device

    quantizer = MidiQuantizer(
        n_dstart_bins=cfg.quantization.dstart_bin,
        n_duration_bins=cfg.quantization.duration_bin,
        n_velocity_bins=cfg.quantization.velocity_bin,
    )
    tokenizer = QuantizedMidiEncoder(
        dstart_bin=cfg.quantization.dstart_bin,
        duration_bin=cfg.quantization.duration_bin,
        velocity_bin=cfg.quantization.velocity_bin,
    )

    model_config = RobertaConfig(vocab_size=tokenizer.vocab_size)
    model = RobertaForMaskedLM(model_config).to(device)

    model.load_state_dict(checkpoint["model"])
    # model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset_name = "JasiekKaczmarczyk/maestro-v1-sustain-masked"

    queries_txt = st.text_input(label="Queries", value="chopin;mozart")
    queries = str.split(queries_txt, sep=";")

    acc = evaluate_classification(
        model, dataset_name=dataset_name, quantizer=quantizer, tokenizer=tokenizer, queries=queries, device=device
    )
    st.write(f"Classification accuracy is: {acc}")


if __name__ == "__main__":
    main()
