import itertools

import numpy as np
import pandas as pd

from data.masking import AwesomeMasks


class MidiEncoder:
    def tokenize(self, record: dict) -> list[str]:
        raise NotImplementedError("Your encoder needs *tokenize* implementation")

    def untokenize(self, tokens: list[str]) -> pd.DataFrame:
        raise NotImplementedError("Your encoder needs *untokenize* implementation")

    def decode(self, token_ids: list[int]) -> pd.DataFrame:
        tokens = [self.vocab[token_id] for token_id in token_ids]
        df = self.untokenize(tokens)

        return df

    def encode(self, record: dict) -> np.ndarray:
        tokens = self.tokenize(record)
        token_ids = np.array([self.token_to_id[token] for token in tokens])
        return token_ids


class QuantizedMidiEncoder(MidiEncoder):
    def __init__(self, dstart_bin: int, duration_bin: int, velocity_bin: int):
        self.dstart_bin = dstart_bin
        self.duration_bin = duration_bin
        self.velocity_bin = velocity_bin

        self.keys = ["pitch", "dstart_bin", "duration_bin", "velocity_bin"]
        self.specials = ["<cls>", "<mask>"]
        self.mask_tokens = AwesomeMasks().vocab

        # ... and add midi tokens
        self.vocab = self._build_vocab()
        self.token_to_id = {token: it for it, token in enumerate(self.vocab)}

    def __rich_repr__(self):
        yield "QuantizedMidiEncoder"
        yield "vocab_size", self.vocab_size

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def _build_vocab(self):
        vocab = list(self.specials)
        vocab += self.mask_tokens

        src_iterators_product = itertools.product(
            # Always include 88 pitches
            range(21, 109),
            range(self.dstart_bin),
            range(self.duration_bin),
            range(self.velocity_bin),
        )

        for pitch, dstart, duration, velocity in src_iterators_product:
            key = f"{pitch}-{dstart}-{duration}-{velocity}"
            vocab.append(key)

        return vocab

    def tokenize(self, record: dict) -> list[str]:
        # TODO I don't love the idea of adding tokens durint *tokenize* call
        # If we want to pretend that our midi sequences have start and finish
        # we should take care of that before we get here :alarm:

        # extract features as np.arrays
        pitch = record["pitch"]
        dstart_bin = record["dstart_bin"]
        duration_bin = record["duration_bin"]
        velocity_bin = record["velocity_bin"]

        n_samples = len(pitch)

        tokens = [f"{pitch[i]}-{dstart_bin[i]}-{duration_bin[i]}-{velocity_bin[i]}" for i in range(n_samples)]

        return tokens

    def untokenize(self, tokens: list[str]) -> pd.DataFrame:
        samples = []
        for token in tokens:
            if token in self.specials:
                continue

            values_txt = token.split("-")
            values = [eval(txt) for txt in values_txt]
            samples.append(values)

        df = pd.DataFrame(samples, columns=self.keys)

        return df


# if __name__ == "__main__":
#     import numpy as np

#     quantization_cfg = DictConfig({
#         "dstart": 7,
#         "duration": 7,
#         "velocity": 7,
#     })

#     record = {
#         "pitch": np.array([35, 88, 27, 102, 55]),
#         "dstart_bin": np.array([4, 3, 0, 1, 2]),
#         "duration_bin": np.array([4, 3, 0, 1, 2]),
#         "velocity_bin": np.array([4, 3, 0, 1, 2]),
#         "masked": [False, True, False, True, False],
#     }

#     tokenizer = QuantizedMidiEncoder(quantization_cfg)

#     tokens = tokenizer.encode(record)

#     print(tokens)
