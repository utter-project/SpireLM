from os.path import join, basename, splitext, exists
from functools import partial
import yaml

import soundfile as sf
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler, SequentialSampler, RandomSampler, BatchSampler, DataLoader
from datasets import load_from_disk, load_dataset, Audio


class AudioTSVDataset(Dataset):

    def __init__(self, tsv_path=None, examples=None):
        assert (tsv_path is not None) != (examples is not None)
        if tsv_path is not None:
            with open(tsv_path) as f:
                root = f.readline().rstrip()
                lines = [line.rstrip().split("\t") for line in f]
            self.data = [(join(root, wav), int(n_samples)) for (wav, n_samples) in lines]
        else:
            self.data = examples

    def select(self, indices):
        # return a new dataset containing the relevant indices from self
        selected = [self.data[i] for i in indices]
        return AudioTSVDataset(examples=selected)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, n_samples = self.data[idx]
        return {"n_samples": n_samples, "audio_path": path, "idx": idx}


class SafeAudioDataset(Dataset):
    def __init__(
        self,
        hf_dataset,
        placeholder_len=0,
        placeholder_sr=16000
    ):
        """
        hf_dataset: Hugging Face Dataset
        decode_from_path: True if using Audio(decode=False)
        """
        self.ds = hf_dataset
        self.placeholder_len = placeholder_len
        self.placeholder_sr = placeholder_sr

    def __len__(self):
        return len(self.ds)

    def _placeholder(self):
        return {
            "array": np.zeros(self.placeholder_len, dtype=np.float32),
            "sampling_rate": self.placeholder_sr,
        }

    def __getitem__(self, idx):
        try:
            ex = self.ds[idx]
            return ex
        except sf.LibsndfileError:
            ex = dict()
            ex["idx"] = idx  # this assumes the dataset keys correspond to the index column. Seems like a safe assumption?
            ex["audio"] = self._placeholder()
            return ex


def collate_fn(inputs, feature_extractor):
    audios = [sf.read(inp["audio_path"])[0] for inp in inputs]
    batch = feature_extractor(
        audios,
        sampling_rate=feature_extractor.sampling_rate,
        return_tensors="pt",
        padding=True,
        return_attention_mask=True
    )
    batch["indices"] = [inp["idx"] for inp in inputs]
    if "seconds" not in inputs[0]:
        batch["seconds"] = [audio.shape[0] / feature_extractor.sampling_rate for audio in audios]
    return batch


def collate_hf(inputs, feature_extractor):
    audios = [inp["audio"]["array"] for inp in inputs]
    batch = feature_extractor(
        audios,
        sampling_rate=feature_extractor.sampling_rate,
        return_tensors="pt",
        padding=True,
        return_attention_mask=True
    )
    batch["indices"] = [inp["idx"] for inp in inputs]
    if "seconds" not in inputs[0]:
        batch["seconds"] = [audio.shape[0] / feature_extractor.sampling_rate for audio in audios]
    return batch


class LengthKeySortedAudioSampler(Sampler):
    """
    This is probably the worst possible way to do it
    """

    def __init__(self, data, length_key="seconds"):
        self.data = data
        self.length_key = length_key

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        # this is very bad with a hugging face dataset.
        # And also
        sorted_data = sorted(self.data, key=lambda x: x[self.length_key], reverse=True)
        for ex in sorted_data:
            yield ex["idx"]


class LengthSortedAudioSampler(Sampler):
    """
    This should be better than LengthKeySortedAudioSampler because it doesn't
    need to access the HF dataset at all
    """

    def __init__(self, indices, lengths):
        self.indices = indices
        # self.lengths = lengths

        self.index_order = np.argsort(lengths)
        self.descending_lengths = lengths[self.index_order]

    def __len__(self):
        return len(self.indices)

    def __iter__(self):
        for i in self.index_order:
            yield self.indices[i]


class NaiveTokenBatchSampler(BatchSampler):

    def __init__(self, data: AudioTSVDataset, sampler: Sampler, batch_size: int, drop_last: bool = False, length_key: str = "seconds"):
        super().__init__(sampler, batch_size, drop_last)
        self.data = data
        self.length_key = length_key

    def __iter__(self):
        batch = []
        longest_len = 0

        # iterate over examples in the order dictated by the sampler
        for idx in self.sampler:
            ex = self.data[idx]

            n_tokens = ex[self.length_key]
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


class TokenBatchSampler(BatchSampler):

    def __init__(self, sampler: Sampler, batch_size: int, drop_last: bool = False):
        super().__init__(sampler, batch_size, drop_last)

    def __iter__(self):
        batch = []
        longest_len = 0

        for idx, length in zip(self.sampler, self.sampler.descending_lengths):
            batch.append(idx)
            longest_len = max(longest_len, length)

            if longest_len * len(batch) >= self.batch_size:
                yield batch
                batch = []
                longest_len = 0
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        raise TypeError("The number of batches in a token-batched epoch is not known in advance")


def load_hf_audio_dataset(
        path, path_extra="", split="train",
        resample_to=None, from_disk=True,
        start_ix=0, n_examples=0, remove_audio=False, add_index=False, filter_mic=None):

    if from_disk:
        if "openslr/librispeech_asr" in path:
            assert path_extra
            split = split + "." + path_extra
        elif path_extra:
            path = join(path, path_extra)
        dataset = load_from_disk(path)[split]
    else:
        if path_extra:
            dataset = load_dataset(path, path_extra, split=split)
        else:
            dataset = load_dataset(path, split=split)

    if remove_audio:
        dataset = dataset.remove_columns("audio")
    elif resample_to is not None:
        dataset = dataset.cast_column("audio", Audio(sampling_rate=resample_to))

    dataset = dataset.skip(start_ix)
    if n_examples > 0:
        examples_to_take = min(len(dataset), n_examples)
        dataset = dataset.take(examples_to_take)

    if add_index:
        dataset = dataset.add_column(name="idx", column=list(range(len(dataset))))

    if path == "CSTR-Edinburgh/vctk":
        # this is hardcoding, but it's a salient fact for a useful dataset.
        audio_paths = dataset.remove_columns("audio")["file"]
        mic = [splitext(basename(audio_path))[0].split("_")[-1] for audio_path in audio_paths]
        dataset = dataset.add_column(name="mic", column=mic)
        if filter_mic is not None:
            dataset = dataset.filter(lambda ex: ex["mic"] == filter_mic)

    return dataset


def _load_dataset_from_config(config):
    with open(config) as f:
        dataset_config = yaml.safe_load(f)
    path = dataset_config["path"]  # only required field
    config = dataset_config.get("config", None)
    split = dataset_config.get("split", None)
    filter_mic = dataset_config.get("filter_mic", None)  # only for VCTK

    # config-based smart dataset load
    if exists(path) and exists(join(path, "dataset_info.json")):
        # hf-disk case
        dataset = load_from_disk(path)[split]
    elif exists(path) and path.endswith(".tsv"):
        return AudioTSVDataset(tsv_path=path), True
    else:
        # hf-cache case
        if config:
            dataset = load_dataset(path, config, split=split)
        else:
            dataset = load_dataset(path, split=split)

    if filter_mic is not None:
        audio_paths = dataset.remove_columns("audio")["file"]
        mic = [splitext(basename(audio_path))[0].split("_")[-1] for audio_path in audio_paths]
        dataset = dataset.add_column(name="mic", column=mic)
        if filter_mic is not None:
            dataset = dataset.filter(lambda ex: ex["mic"] == filter_mic)

    return dataset, False


def _postprocess_dataset(
        dataset, resample_to=None, start_ix=0, n_examples=0, remove_audio=False, add_index=False):

    if remove_audio:
        dataset = dataset.remove_columns("audio")
    elif resample_to is not None:
        dataset = dataset.cast_column("audio", Audio(sampling_rate=resample_to))

    dataset = dataset.skip(start_ix)
    if n_examples > 0:
        examples_to_take = min(len(dataset), n_examples)
        dataset = dataset.take(examples_to_take)

    if add_index:
        dataset = dataset.add_column(name="idx", column=list(range(len(dataset))))
    return dataset


def load_audio_dataset(
        config, resample_to=None, start_ix=0, n_examples=0, remove_audio=False,
        add_index=False):

    dataset, is_tsv = _load_dataset_from_config(config)
    dataset = _postprocess_dataset(
        dataset,
        resample_to=resample_to,
        start_ix=start_ix,
        n_examples=n_examples,
        remove_audio=remove_audio,
        add_index=add_index)
    return dataset, is_tsv


class AdaptiveWeightedMultiDataLoader:
    def __init__(self, loaders, target_weights=None, alpha=0.02):
        self.loaders = loaders
        if target_weights is not None:
            self.target_weights = torch.tensor(target_weights, dtype=torch.float)
        else:
            # if unspecified, use uniform weights
            self.target_weights = torch.ones(len(loaders))
        self.target_weights /= self.target_weights.sum()

        self.alpha = alpha

        # initialize duration estimates
        self.avg_lengths = torch.ones(len(loaders))

    def _sampling_probs(self):
        probs = self.target_weights / self.avg_lengths
        return probs / probs.sum()

    def __iter__(self):
        iters = [iter(l) for l in self.loaders]

        while True:
            probs = self._sampling_probs()
            idx = torch.multinomial(probs, 1).item()

            try:
                batch = next(iters[idx])
            except StopIteration:
                iters[idx] = iter(self.loaders[idx])
                batch = next(iters[idx])

            # assume batch contains audio tensors
            # compute batch average duration
            lengths = batch["seconds"]
            # batch_avg = lengths.float().mean()
            batch_avg = sum(lengths) / len(lengths)

            # EMA update
            self.avg_lengths[idx] = (
                (1 - self.alpha) * self.avg_lengths[idx]
                + self.alpha * batch_avg
            )

            yield batch


def _build_single_dataloader(
        config,
        feature_extractor, num_workers=0, batch_size=1, start_ix=0,
        n_examples=0,
        resample_to=None, shuffle=False, torch_random=None, pin_memory=False,
        token_batching=False, example_lengths=None, collator=None, placeholder_len=0):

    # Build dataset
    dataset, is_tsv = load_audio_dataset(
        config,
        add_index=True,
        start_ix=start_ix,
        n_examples=n_examples,
        resample_to=resample_to
    )

    if example_lengths is not None:
        lengths = np.load(example_lengths)[start_ix: start_ix + n_examples]
        print("Average length of this shard", lengths.mean())
    else:
        lengths = None

    length_before_validating = len(dataset)
    print("Dataset length: {}".format(length_before_validating))

    # build the collator
    if collator is None:
        # if no custom collator is provided, use one of the ones defined in this file
        collate_func = collate_fn if is_tsv else collate_hf
        collator = partial(collate_func, feature_extractor=feature_extractor)

    if token_batching:
        if is_tsv:
            sampler = LengthKeySortedAudioSampler(dataset, length_key="n_samples")
            batch_sampler = NaiveTokenBatchSampler(dataset, sampler, batch_size, length_key="n_samples")
        else:
            sampler = LengthSortedAudioSampler(dataset["idx"], lengths)
            batch_sampler = TokenBatchSampler(sampler, batch_size)
    else:
        sampler = RandomSampler(dataset, generator=torch_random) if shuffle else SequentialSampler(dataset)
        batch_sampler = BatchSampler(sampler, batch_size=batch_size, drop_last=False)

    if not is_tsv:
        # in theory, this should make it possible to handle missing audio gracefully
        dataset = SafeAudioDataset(dataset, placeholder_len=placeholder_len)
    loader = DataLoader(
        dataset,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collator,
        batch_sampler=batch_sampler
    )

    # count number of batches
    if token_batching:
        n_batches = len([b for b in batch_sampler])
    else:
        n_batches = len(loader)

    return loader, n_batches, length_before_validating


def build_dataloader(
        config,
        feature_extractor,
        dataset_weights=None, num_workers=0, batch_size=1, start_ix=0, n_examples=0,
        resample_to=None, shuffle=False, torch_random=None, pin_memory=False,
        token_batching=False, example_lengths=None, collator=None, placeholder_len=0):

    if isinstance(config, str):
        config = [config]

    loaders = []
    n_batches = []
    pre_valid_length = []

    for cf in config:
        single_loader, single_n_batches, single_length_before_batching = _build_single_dataloader(
            config=cf,
            feature_extractor=feature_extractor,
            num_workers=num_workers,
            batch_size=batch_size,
            start_ix=start_ix,
            n_examples=n_examples,
            resample_to=resample_to,
            shuffle=shuffle,
            torch_random=torch_random,
            pin_memory=False,
            token_batching=token_batching,
            example_lengths=example_lengths,
            collator=collator,
            placeholder_len=placeholder_len
        )
        loaders.append(single_loader)
        n_batches.append(single_n_batches)
        pre_valid_length.append(single_length_before_batching)

    if len(config) == 1:
        return loaders[0], n_batches[0], pre_valid_length[0]

    weighted_loader = AdaptiveWeightedMultiDataLoader(loaders, dataset_weights)
    return weighted_loader, sum(n_batches), sum(pre_valid_length)
