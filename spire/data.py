from os.path import join
from functools import partial

from tqdm import tqdm
import soundfile as sf
from torch.utils.data import Dataset, Sampler, BatchSampler, DataLoader
from transformers import Wav2Vec2FeatureExtractor
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


def load_hf_audio_dataset(
        path, path_extra="", split="train",
        resample_to=None, from_disk=True,
        start_ix=0, n_examples=0, remove_audio=False, add_index=False):

    if remove_audio:
        assert resample_to is None

    if from_disk:
        if "openslr/librispeech_asr" in path:
            assert path_extra
            split = split + "." + path_extra
        elif path_extra:
            path = join(path, path_extra)
        dataset = load_from_disk(path)[split]
    else:
        # currently just VCTK
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

    return dataset


def get_valid_indices(dataset):

    ix = []

    if isinstance(dataset, AudioTSVDataset):
        for i in tqdm(range(len(dataset))):
            ex = dataset[i]
            if ex["n_samples"] == 0:
                continue
            ix.append(i)

    else:
        for i in tqdm(range(len(dataset))):
            try:
                _ = dataset[i]
            except sf.LibsndfileError:
                continue
            ix.append(i)
    return ix


def build_dataloader(
        path, num_workers=0, batch_size=1, dataset_type="tsv", start_ix=0,
        n_examples=0, validate_examples=False, path_extra="en", hf_location="disk",
        hf_split="test", resample_to=None):

    # Build dataset
    if dataset_type == "tsv":
        dataset = AudioTSVDataset(tsv_path=path)
    else:
        dataset = load_hf_audio_dataset(
            path, path_extra=path_extra, resample_to=resample_to, split=hf_split,
            from_disk=hf_location == "disk", add_index=True,
            start_ix=start_ix, n_examples=n_examples
        )

    # optionally filter invalid examples
    length_before_validating = len(dataset)
    if validate_examples:
        valid_indices = get_valid_indices(dataset)
        dataset = dataset.select(valid_indices)
    print("Dataset lengths:")
    print("Raw: {}\tAfter validating: {}".format(length_before_validating, len(dataset)))

    # Build the loader
    feature_extractor = Wav2Vec2FeatureExtractor()
    if dataset_type == "tsv":
        sampler = LengthSortedAudioSampler(dataset)
        batch_sampler = TokenBatchSampler(dataset, sampler, batch_size)
        n_batches = len([b for b in batch_sampler])
        collate_func = collate_fn
        loader_kwargs = {"batch_sampler": batch_sampler}
    else:
        n_batches = None
        collate_func = collate_hf
        loader_kwargs = {
            "drop_last": False, "batch_size": batch_size
        }
    collator = partial(collate_func, feature_extractor=feature_extractor)
    loader = DataLoader(
        dataset, num_workers=num_workers, pin_memory=True, collate_fn=collator, **loader_kwargs
    )

    if n_batches is None:
        n_batches = len(loader)
    return loader, n_batches, length_before_validating
