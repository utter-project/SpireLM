from os.path import join
from functools import partial

import soundfile as sf
from torch.utils.data import Dataset, Sampler, BatchSampler, DataLoader
from transformers import Wav2Vec2FeatureExtractor
from datasets import load_from_disk, Audio


class AudioTSVDataset(Dataset):

    def __init__(self, tsv_path, sample_rate=16000):
        with open(tsv_path) as f:
            root = f.readline().rstrip()
            lines = [line.rstrip().split("\t") for line in f]
        self.data = [(join(root, wav), int(n_samples)) for (wav, n_samples) in lines]
        # self.data = sorted(self.data, key=lambda x: x[1], reverse=True)
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # this is the one that actually loads the stuff
        path, n_samples = self.data[idx]
        # do we really want to do the reading here?
        # waveform, sample_rate = sf.read(path, dtype="float32")
        # assert sample_rate == self.sample_rate
        # return {"audio": waveform, "n_samples": n_samples, "audio_path": path, "idx": idx}
        return {"n_samples": n_samples, "audio_path": path, "idx": idx}


def collate_fn(inputs, feature_extractor):
    # inputs should be a list of dicts returned from AudioTSVDataset
    # audios = [inp["audio"] for inp in inputs]
    audios = [sf.read(inp["audio_path"])[0] for inp in inputs]
    batch = feature_extractor(
        audios,
        sampling_rate=feature_extractor.sampling_rate,
        return_tensors="pt",
        padding=True,
        return_attention_mask=True
    )
    batch["indices"] = [inp["idx"] for inp in inputs]
    return batch


def collate_hf(inputs, feature_extractor):
    # inputs should be a list of dicts returned from AudioTSVDataset
    # audios = [inp["audio"] for inp in inputs]
    audios = [inp["audio"]["array"] for inp in inputs]
    batch = feature_extractor(
        audios,
        sampling_rate=feature_extractor.sampling_rate,
        return_tensors="pt",
        padding=True,
        return_attention_mask=True
    )
    # batch["indices"] = [inp["idx"] for inp in inputs]
    return batch


class LengthSortedAudioSampler(Sampler):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        sorted_data = sorted(self.data, key=lambda x: x["n_samples"], reverse=True)
        for ex in sorted_data:
            yield ex["idx"]


# is there any actual reason to inherit?
class TokenBatchSampler(BatchSampler):

    def __init__(self, data: AudioTSVDataset, sampler: Sampler, batch_size: int, drop_last: bool = False):
        super().__init__(sampler, batch_size, drop_last)
        self.data = data

    def __iter__(self):
        batch = []
        longest_len = 0

        # iterate over examples in the order dictated by the sampler
        for idx in self.sampler:
            ex = self.data[idx]

            #
            n_tokens = ex["n_samples"]
            batch.append(idx)
            longest_len = max(longest_len, n_tokens)

            if longest_len * len(batch) >= self.batch_size:
                yield batch
                batch = []
                longest_len = 0
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        raise TypeError("The number of batches in a token-batched epoch is not known in advance")


def load_commonvoice(path, lang="en", split="train"):
    dataset = load_from_disk(join(path, lang))[split]
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    return dataset


def load_gigaspeech(path, size="xl", split="train"):

    dataset = load_from_disk(join(path, size))[split]
    # disable_caching()
    # dataset = dataset.map(truecase_and_fix_punct)
    return dataset


def load_spgi(path, split="train"):
    """
    Identical to the commonvoice one besides the path parsing
    """
    dataset = load_from_disk(path)[split]
    return dataset


# todo: abstract this out for HF datasets
def build_dataloader(path, sample_rate=16000, num_workers=0, batch_size=1, dataset_type="tsv"):

    feature_extractor = Wav2Vec2FeatureExtractor()
    if dataset_type == "tsv":
        dataset = AudioTSVDataset(path, sample_rate=sample_rate)
        sampler = LengthSortedAudioSampler(dataset)
        batch_sampler = TokenBatchSampler(dataset, sampler, batch_size)
        n_batches = len([b for b in batch_sampler])
        loader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=partial(collate_fn, feature_extractor=feature_extractor),
            num_workers=num_workers,
            pin_memory=True,
            # drop_last=False
        )
        return loader, n_batches
    else:
        # huggingface dataset. How about the dataset_type is "commonvoice", "spgi",
        # or "gigaspeech"?
        if dataset_type == "commonvoice":
            dataset = load_commonvoice(path, split="train")
        elif dataset_type == "spgi":
            dataset = load_spgi(path)
        else:
            dataset = load_gigaspeech(path)

        loader = DataLoader(
            dataset,
            num_workers=num_workers,
            pin_memory=True,
            batch_size=batch_size,
            drop_last=False,
            collate_fn=partial(collate_hf, feature_extractor=feature_extractor)
        )
        # does it *have* a length?
        return loader, len(loader)
