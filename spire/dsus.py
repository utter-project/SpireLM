"""
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import unicodedata
from os.path import join
from functools import partial

from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import soundfile as sf

from spire.kmeans import KmeansForInference
from spire.hubert_features import HFHubertFeatureReader


PRIVATE_OFFSET = 983040


def extra_id(i):
    return "<extra_id_{}>".format(str(i))


def pua(i):
    private_char = chr(int(i) + PRIVATE_OFFSET)
    assert unicodedata.category(private_char) == "Co"  # otherwise it's not private
    return private_char


def indices2dsus(indices, dsu_format="pua"):
    dsu_formatter = pua if dsu_format == "pua" else extra_id
    return "".join([dsu_formatter(i) for i in indices])


def dedup(tokens):
    out_tokens = []
    last_token = None
    for t in tokens:
        if t != last_token:
            out_tokens.append(t)
            last_token = t
    return out_tokens


class AudioTSVDataset(Dataset):

    def __init__(self, tsv_path, sample_rate=16000):
        with open(tsv_path) as f:
            root = f.readline().rstrip()
            lines = [line.rstrip().split("\t") for line in f]
        self.data = [(join(root, wav), n_samples) for (wav, n_samples) in lines]
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
    inputs = feature_extractor(audios, sampling_rate=feature_extractor.sampling_rate, return_tensors="pt", padding=True, return_attention_mask=True)
    return inputs


# this whole thing needs to be refactored, I think. And that's fine.
# but also, the feature-reading/kmeans could be refactored into a single
# nn.Module, I think. Pass ckpt_path, layer, and kmeans path. Forward generates
# scores for each cluster, and you implement a predict method as well that picks
# the closest cluster.
class Labeler:

    def __init__(
        self, ckpt_path, km_path, feature_layer=22, max_chunk=1600000,
        deduplicated=True, dsu_format="pua", kmeans_device="cuda:0", legacy_audio=False
    ):
        self.hubert = HFHubertFeatureReader(ckpt_path, feature_layer)
        self.kmeans = KmeansForInference(km_path, device=kmeans_device)
        self.deduplicated = deduplicated
        self.dsu_format = dsu_format

    def _get_features(self, path=None, batch=None, nsample=None, attention_mask=None):
        return self.hubert.get_feats(path=path, batch=batch, ref_len=nsample, attention_mask=attention_mask)

    def _get_indices(self, feats, output_mask=None):
        labels = self.kmeans(feats).masked_fill_(~output_mask, -1)
        # labels = self.kmeans(feats).to("cpu")
        labels = labels.to("cpu").tolist()  # should be a list of lists

        # let's do this with tensors instead: masked_fill_ is the easiest way
        # labels = [lab[:output_length] for lab, output_length in zip(labels, output_lengths)]
        if self.deduplicated:
            labels = [dedup([l for l in lab if l != -1]) for lab in labels]
        return labels

    def label(self, path=None, batch=None, indices_only=False, attention_mask=None):
        feats = self._get_features(path=path, batch=batch, attention_mask=attention_mask)  # b x len x dim
        # here we can call
        output_mask = self.hubert.model._get_feature_vector_attention_mask(feats.shape[1], attention_mask)

        indices = self._get_indices(feats, output_mask=output_mask)
        if indices_only:
            return indices
        labels = [indices2dsus(ix) for ix in indices]
        return labels

    def label_corpus(self, tsv_path, sample_rate=16000, indices_only=False, batch_size=1, num_workers=1):
        dataset = AudioTSVDataset(tsv_path, sample_rate=sample_rate)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=partial(collate_fn, feature_extractor=self.hubert.fe),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False
        )
        # make the dataloader...
        # iterate through batches (will need to provide batch size as well)
        #
        labels = []
        for batch in tqdm(loader):
            inp = batch.input_values.cuda()
            mask = batch.attention_mask.cuda()  # should it go to cuda here already?
            input_lengths = mask.sum(dim=-1)
            output_lengths = self.hubert.model._get_feat_extract_output_lengths(input_lengths)
            batch_labels = self.label(batch=inp, indices_only=indices_only, attention_mask=mask)
            labels.extend(batch_labels)  # right type?
        return labels
