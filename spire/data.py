from os.path import join

import soundfile as sf
from torch.utils.data import Dataset


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
        waveform, sample_rate = sf.read(path, dtype="float32")
        assert sample_rate == self.sample_rate
        return {"audio": waveform, "n_samples": n_samples, "audio_path": path}


def collate_fn(inputs, feature_extractor):
    # inputs should be a list of dicts returned from AudioTSVDataset
    audios = [inp["audio"] for inp in inputs]
    inputs = feature_extractor(
        audios,
        sampling_rate=feature_extractor.sampling_rate,
        return_tensors="pt",
        padding=True,
        return_attention_mask=True
    )
    return inputs
