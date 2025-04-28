"""
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import unicodedata

import torch

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


class Labeler:

    def __init__(
        self, ckpt_path, km_path, feature_layer=22, max_chunk=1600000,
        deduplicated=True, dsu_format="pua", kmeans_device="cuda:0", kmeans_backend="torch", legacy_audio=False
    ):
        self.hubert = HFHubertFeatureReader(ckpt_path, feature_layer)
        self.kmeans = KmeansForInference(km_path, device=kmeans_device, backend=kmeans_backend)
        self.deduplicated = deduplicated
        self.dsu_format = dsu_format

    def _get_features(self, wav_path, nsample=None):
        return self.hubert.get_feats(wav_path, nsample)

    def _get_indices(self, feats):
        labels = self.kmeans(feats)
        if isinstance(labels, torch.Tensor):
            labels = labels.to("cpu")
        labels = labels.tolist()
        if self.deduplicated:
            labels = dedup(labels)
        return labels

    def label(self, wav_path, indices_only=False):
        feats = self._get_features(wav_path)
        indices = self._get_indices(feats)
        if indices_only:
            return indices
        labels = indices2dsus(indices)
        return labels
