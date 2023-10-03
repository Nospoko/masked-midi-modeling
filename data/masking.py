import numpy as np


class Mask:
    token: str

    def mask(self, record: dict):
        """
        Args:
            record (dict): should contain keys: pitch, dstart, duration, velocity, dstart_bin, duration_bin, velocity_bin all as numpy array
        """
        raise NotImplementedError("Subclass must implement this method")

    def masking_space(self, record: dict):
        raise NotImplementedError("Subclass must implement this method")


class RandomMask(Mask):
    token: str = "<Random Mask>"

    def mask(self, record: dict, p: float) -> np.ndarray:
        noise = np.random.random(len(record["pitch"]))
        masked = noise <= p

        return masked

    def masking_space(self, record: dict) -> np.ndarray:
        return record["pitch"] > 0


class LeftHandMask(Mask):
    token: str = "<LH Mask>"

    def mask(self, record: dict, p: float) -> np.ndarray:
        assert p < 0.5, "This strategy targets only a half on available notes, not possible to mask more than 0.5"
        record_len = len(record["pitch"])

        n_masked = np.random.binomial(record_len, p)
        n_masked = min(n_masked, record_len // 2)

        ids = record["masking_spaces"][self.token]
        to_mask = np.random.choice(np.arange(record_len)[ids], size=n_masked, replace=False)
        masked = np.full_like(record["pitch"], fill_value=False, dtype=bool)
        masked[to_mask] = True

        return masked

    def masking_space(self, record: dict) -> np.ndarray:
        middle_pitch = np.median(record["pitch"])
        ids = record["pitch"] <= middle_pitch
        return ids


class RightHandMask(Mask):
    token: str = "<RH Mask>"

    def mask(self, record: dict, p: float) -> np.ndarray:
        assert p < 0.5, "This strategy targets only a half on available notes, not possible to mask more than 0.5"
        record_len = len(record["pitch"])

        n_masked = np.random.binomial(record_len, p)
        n_masked = min(n_masked, record_len // 2)

        ids = record["masking_spaces"][self.token]
        to_mask = np.random.choice(np.arange(record_len)[ids], size=n_masked, replace=False)
        masked = np.full_like(record["pitch"], fill_value=False, dtype=bool)
        masked[to_mask] = True

        return masked

    def masking_space(self, record: dict) -> np.array:
        middle_pitch = np.median(record["pitch"])
        ids = record["pitch"] >= middle_pitch
        return ids


class HarmonicRootMask(Mask):
    token: str = "<Harmonic Root Mask>"

    def mask(self, record: dict, p: float) -> np.ndarray:
        ids = record["masking_spaces"][self.token]
        record_len = len(record["pitch"])

        n_masked = min(np.random.binomial(record_len, p), ids.sum())
        to_mask = np.random.choice(np.arange(record_len)[ids], size=n_masked, replace=False)
        masked = np.full_like(record["pitch"], fill_value=False, dtype=bool)
        masked[to_mask] = True

        return masked

    def masking_space(self, record: dict) -> np.ndarray:
        # masking space will be 3 most popular pitches
        absolute_pitch = record["pitch"] % 12

        counts = np.bincount(absolute_pitch)
        top_k = np.argsort(-counts)[:3]
        ids = np.isin(absolute_pitch, top_k)
        return ids


class HarmonicOutliersMask(Mask):
    token: str = "<Harmonic Outliers Mask>"

    def mask(self, record: dict, p: float) -> np.ndarray:
        ids = record["masking_spaces"][self.token]
        record_len = len(record["pitch"])

        n_masked = min(np.random.binomial(record_len, p), ids.sum())
        to_mask = np.random.choice(np.arange(record_len)[ids], size=n_masked, replace=False)
        masked = np.full_like(record["pitch"], fill_value=False, dtype=bool)
        masked[to_mask] = True

        return masked

    def masking_space(self, record: dict) -> dict:
        # masking space will be 3 least popular pitches
        absolute_pitch = record["pitch"] % 12

        counts = np.bincount(absolute_pitch)
        # if some pitch was counted 0 times, 100 is added to prevent non existent pitches from being counted as top_k
        counts = np.where(counts == 0, 100, counts)
        top_k = np.argsort(counts)[:3]
        ids = np.isin(absolute_pitch, top_k)
        return ids


class AwesomeMasks:
    def __init__(self):
        self.masks = [
            RandomMask(),
            LeftHandMask(),
            RightHandMask(),
            HarmonicRootMask(),
            HarmonicOutliersMask(),
        ]

    def __rich_repr__(self):
        yield "AwesomeMasks",
        yield "n_masks", len(self.masks)

    def apply(self, record: dict, p: float) -> tuple[np.ndarray, str]:
        mask = np.random.choice(self.masks)
        masked = mask.mask(record, p)

        return masked, mask.token

    @property
    def vocab(self) -> list[str]:
        vocab = [mask.token for mask in self.masks]
        return vocab
