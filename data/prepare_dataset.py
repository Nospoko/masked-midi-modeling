import os
import argparse
import json

import fortepyan as ff
from tqdm import tqdm
from datasets import Value, Dataset, Features, Sequence, DatasetDict, load_dataset

from data.masking import AwesomeMasks

def process_dataset(dataset: Dataset, sequence_len: int, sequence_step: int, masks: AwesomeMasks) -> list[dict]:
    processed_records = []

    for record in tqdm(dataset, total=dataset.num_rows):
        # print(record)
        piece = ff.MidiPiece.from_huggingface(record)
        processed_record = process_record(piece, sequence_len, sequence_step, masks)

        processed_records += processed_record

    return processed_records


def process_record(piece: ff.MidiPiece, sequence_len: int, sequence_step: int, masks: AwesomeMasks) -> list[dict]:
    piece.df["next_start"] = piece.df.start.shift(-1)
    piece.df["dstart"] = piece.df.next_start - piece.df.start
    piece.df["dstart"] = piece.df["dstart"].fillna(0)

    midi_filename = piece.source['midi_filename']

    record = []

    n_samples = 1 + (piece.size - sequence_len) // sequence_step
    for it in range(n_samples):
        start = it * sequence_step
        finish = start + sequence_len
        part = piece[start:finish]

        sequence = {
            "midi_filename": midi_filename,
            "source": json.dumps(part.source),
            "pitch": part.df.pitch.astype("int16").values,
            "dstart": part.df.dstart.astype("float32").values,
            "duration": part.df.duration.astype("float32").values,
            "velocity": part.df.velocity.astype("int16").values,
        }

        masking_spaces = {}

        for mask in masks.masks:
            # add new keys to sequence corresponding to masking space for each mask type
            masking_spaces[mask.token] = mask.masking_space(sequence)
        
        sequence["masking_spaces"] = masking_spaces

        record.append(sequence)

    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dataset_path", type=str, default="roszcz/giant-midi-sustain")
    parser.add_argument("--sequence_len", type=int, default=128)
    parser.add_argument("--target_dataset_path", type=str, default="JasiekKaczmarczyk/giant-midi-sustain")
    args = parser.parse_args()


    # get huggingface token from environment variables
    token = os.environ["HUGGINGFACE_TOKEN"]

    dataset = load_dataset(args.source_dataset_path)

    # if no train/val/test splits
    if len(dataset) == 3:
        train_dataset = dataset["train"]
        val_dataset = dataset["validation"]
        test_dataset = dataset["test"]
    elif len(dataset) == 1:
        train_val_test_dataset = dataset["train"].train_test_split(test_size=0.2)
        val_test_dataset = train_val_test_dataset["test"].train_test_split(test_size=0.5)

        train_dataset = train_val_test_dataset["train"]
        val_dataset = val_test_dataset["train"]
        test_dataset = val_test_dataset["test"]

    else:
        raise KeyError("Dataset split error")
    
    masks = AwesomeMasks()

    # building huggingface dataset
    features = Features({
        "midi_filename": Value(dtype="string"),
        "source": Value(dtype="string"),
        "pitch": Sequence(feature=Value(dtype="int16"), length=args.sequence_len),
        "dstart": Sequence(feature=Value(dtype="float32"), length=args.sequence_len),
        "duration": Sequence(feature=Value(dtype="float32"), length=args.sequence_len),
        "velocity": Sequence(feature=Value(dtype="int16"), length=args.sequence_len),
        "masking_spaces": Features({
            mask.token: Sequence(feature=Value(dtype="bool"), length=args.sequence_len) for mask in masks.masks
        })
    })

    train_records = process_dataset(train_dataset, sequence_len=args.sequence_len, sequence_step=args.sequence_len, masks=masks)
    val_records = process_dataset(val_dataset, sequence_len=args.sequence_len, sequence_step=args.sequence_len, masks=masks)
    test_records = process_dataset(test_dataset, sequence_len=args.sequence_len, sequence_step=args.sequence_len, masks=masks)

    dataset = DatasetDict(
        {
            "train": Dataset.from_list(train_records, features=features),
            "validation": Dataset.from_list(val_records, features=features),
            "test": Dataset.from_list(test_records, features=features),
        }
    )

    # print(dataset["train"])
    dataset.push_to_hub(args.target_dataset_path, token=token)


if __name__ == "__main__":
    main()