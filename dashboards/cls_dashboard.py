import torch
import numpy as np
import streamlit as st
from datasets import load_dataset
from sklearn.decomposition import PCA
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
    batch_size: int,
    num_workers: int,
):
    dataset = load_dataset(dataset_name, split="validation")

    ds = MidiDataset(dataset, quantizer, tokenizer, masking_probability=0.0)

    idx = []

    for query in queries:
        idx_query = [i for i, name in enumerate(ds.dataset["source"]) if str.lower(query) in str.lower(name)]
        idx += idx_query

    ds = Subset(ds, indices=idx)

    # dataloaders
    dataloader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    return dataloader


def accuracy(predicted: np.ndarray, target: np.ndarray):
    return np.sum(np.equal(predicted, target)) / len(target)


def evaluate_classification(
    model: RobertaForMaskedLM,
    dataset_name: str,
    quantizer: MidiQuantizer,
    tokenizer: QuantizedMidiEncoder,
    queries: list[str],
    device: torch.device,
):
    dataloader = preprocess_dataset(
        dataset_name,
        quantizer=quantizer,
        tokenizer=tokenizer,
        queries=queries,
        batch_size=1024,
        num_workers=8,
    )
    query_to_id = {k: i for i, k in enumerate(queries)}

    outputs = []
    labels = []

    # velocity time encoding
    for batch in dataloader:
        with torch.no_grad():
            source = batch["source"]
            input_token_ids = batch["input_token_ids"].to(device)

            # shape [batch_size, seq_len, vocab_size]
            out = model(input_ids=input_token_ids).logits
            cls_embedding = out[:, 0, :]
            outputs.append(cls_embedding)

            for s in source:
                for query in queries:
                    if str.lower(query) in str.lower(s):
                        labels.append(query_to_id[query])

    # shape: [num_sequences, vocab_size]
    X = torch.cat(outputs, dim=0).cpu().numpy()
    y = np.array(labels)

    assert len(X) == len(y)

    # reducing dimensions
    pca = PCA(n_components=128)
    X_reduced = pca.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)

    tree = DecisionTreeClassifier()
    tree.fit(X_train, y_train)

    acc = tree.score(X_test, y_test)

    return acc


def main():
    checkpoint = torch.load(
        "checkpoints/masked-midi-modelling-2023-10-06-10-36-params-87.88M.ckpt"
        # "checkpoints/pianomask_2023_10_08_19_10.pt"
    )

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

    queries_txt = st.text_input(label="Queries")
    queries = str.split(queries_txt, sep=";")

    acc = evaluate_classification(
        model, dataset_name=dataset_name, quantizer=quantizer, tokenizer=tokenizer, queries=queries, device=device
    )
    st.write(f"Classification accuracy is: {acc}")


if __name__ == "__main__":
    main()
