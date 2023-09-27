import random

import torch
import numpy as np
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset

from data.quantizer import MidiQuantizer
from data.tokenizer import MidiEncoder
from data.augmentation import change_speed, pitch_shift
from data.masking import AwesomeMasks


class MidiDataset(Dataset):
    def __init__(
            self, 
            dataset: HFDataset, 
            tokenizer: MidiEncoder, 
            augmentation_probability: float = 0.0, 
            masking_probability: float = 0.15
        ):
        super().__init__()

        self.dataset = dataset
        self.quantizer = MidiQuantizer(7, 7, 7)
        self.tokenizer = tokenizer
        self.masks = AwesomeMasks()

        self.augmentation_probability = augmentation_probability
        self.masking_probability = masking_probability

    def __len__(self):
        return len(self.dataset)
    
    def apply_augmentation(self, record: dict):
        # shift pitch augmentation
        if random.random() < self.augmentation_probability:
            # max shift is octave down or up
            shift = random.randint(1, 12)
            record["pitch"] = pitch_shift(record["pitch"], shift)

        # change tempo augmentation
        if random.random() < self.augmentation_probability:
            record["dstart"], record["duration"] = change_speed(record["dstart"], record["duration"])
            # change bins for new dstart and duration values
            record["dstart_bin"] = np.digitize(record["dstart"], self.quantizer.dstart_bin_edges) - 1
            record["duration_bin"] = np.digitize(record["duration"], self.quantizer.duration_bin_edges) - 1

        return record
    
    def apply_masking(self, token_ids: np.ndarray, record: dict):
        # masking, adds new key to record dict called masked
        masked, _ = self.masks.apply(record, p=self.masking_probability)

        blank_idx = self.tokenizer.token_to_id["<blank>"]
        token_ids[masked] = blank_idx

        return token_ids

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.dataset[index]

        filename = record["midi_filename"]
        # cast list to np array
        record = {k: np.array(v) for k, v in record.items() if k != "midi_filename"}

        # sanity check, replace NaN with 0
        if np.any(np.isnan(record["dstart"])):
            record["dstart"] = np.nan_to_num(record["dstart"], copy=False)

        record = self.apply_augmentation(record)

        # tokenization
        token_ids = self.tokenizer.encode(record)

        source_token_ids = self.apply_masking(token_ids, record)
        tgt_token_ids = token_ids.copy()

        # add cls token at the start of sequence
        cls_token = self.tokenizer.token_to_id["<cls>"]
        source_token_ids = np.insert(source_token_ids, obj=0, values=cls_token)
        tgt_token_ids = np.insert(tgt_token_ids, obj=0, values=cls_token)

        record = {
            "filename": filename,
            "source_token_ids": torch.tensor(source_token_ids, dtype=torch.long),
            "tgt_token_ids": torch.tensor(tgt_token_ids, dtype=torch.long),
        }

        return record

# if __name__ == "__main__":
#     from omegaconf import DictConfig
#     from data.tokenizer import QuantizedMidiEncoder
#     from datasets import load_dataset
#     from torch.utils.data import DataLoader
#     from transformers import RobertaForMaskedLM

#     quantization_cfg = DictConfig({
#         "dstart": 7,
#         "duration": 7,
#         "velocity": 7,
#     })

#     ds = load_dataset("JasiekKaczmarczyk/maestro-sustain-quantized", split="train")

#     tokenizer = QuantizedMidiEncoder(quantization_cfg)

#     dataset = MidiDataset(ds, tokenizer, augmentation_probability=0.1)

#     loader = DataLoader(dataset, batch_size=4)

#     print(next(iter(loader))["source_token_ids"])

#     m = RobertaForMaskedLM()