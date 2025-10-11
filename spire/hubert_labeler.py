import torch
import torch.nn as nn
import joblib
from transformers import HubertModel

from spire.utils import load_wav, detokenize


# this whole thing needs to be refactored, I think. And that's fine.
# but also, the feature-reading/kmeans could be refactored into a single
# nn.Module, I think. Pass ckpt_path, layer, and kmeans path. Forward generates
# scores for each cluster, and you implement a predict method as well that picks
# the closest cluster.
class HubertLabeler(nn.Module):

    def __init__(self, ckpt_path, km_path, layer=22, dtype=torch.float32):
        super().__init__()

        self.model = HubertModel.from_pretrained(ckpt_path, torch_dtype=dtype)
        self.model.encoder.layer_norm = nn.Identity()  # kludge to avoid applying final layer norm
        self.model.encoder.layers = self.model.encoder.layers[:layer]

        self.kmeans = KMeans(km_path, dtype=dtype)

    @property
    def vocab_size(self):
        return self.kmeans.vocab_size

    def forward(self, batch, attention_mask=None):
        """
        Take a batch of inputs, return scores for all V clusters
        """
        feats = self.model(batch, attention_mask=attention_mask).last_hidden_state

        dist = self.kmeans(feats)
        return dist

    def predict(self, batch, attention_mask=None, label_mask=None):
        dist = self(batch, attention_mask=attention_mask)
        # label_mask should be a boolean vector of length V
        if label_mask is not None:
            # label_mask is True for kept parts of the vocab, False elsewhere
            dist.masked_fill_(~label_mask, float("inf"))

        labels = dist.argmin(dim=-1)

        # what's feats.shape[1]? It's the length dimension, so it should be
        # dist.shape[1] as well
        if attention_mask is not None:
            output_mask = self.model._get_feature_vector_attention_mask(dist.shape[1], attention_mask)
            labels.masked_fill_(~output_mask, -1)
        return labels

    def label_wav(self, wav_path, **detok_args):
        # read the audio into a batch
        device = self.kmeans.C.device
        batch = load_wav(
            wav_path, device=device, expected_sample_rate=16000
        )

        # call self.predict (no attention mask because it's a single-element batch)
        labels = self.predict(batch)

        # detokenize
        detokenized_labels = detokenize(labels, **detok_args)

        return detokenized_labels


class KMeans(nn.Module):
    """
    Adapted from https://github.com/facebookresearch/fairseq/blob/main/examples/hubert/simple_kmeans/dump_km_label.py,
    copyright (c) Facebook, Inc. and its affiliates.

    K-means adapted from fairseq. Note that using the torch backend may
    (very rarely) produce slightly different cluster assignments than using
    numpy due to numerical differences.
    """
    def __init__(self, km_path, dtype=torch.float32):
        super().__init__()

        # dtype = torch.float32  # hardcode and we cast
        # load the numpy stuff
        km_model = joblib.load(km_path)
        C = km_model.cluster_centers_.transpose()
        Cnorm = (C ** 2).sum(0, keepdims=True)

        # now make some parameters and load the numpy values into them, I guess

        # device = torch.device(device)
        # do we need to load the dtype this way? Or should we do it when we
        # convert the
        self.C = nn.Parameter(torch.from_numpy(C).to(dtype))
        self.Cnorm = nn.Parameter(torch.from_numpy(Cnorm).to(dtype))

    @property
    def vocab_size(self):
        return self.C.shape[1]

    def forward(self, x):
        """
        x: historically len x dim
        - Change the dimension for sum/argmin to -1
        - Then it should work for b x len x dim, I hope
        """
        # x = x.to(self.C.dtype)
        # self.Cnorm is essentially a bias...
        # version 1:
        dist = x.pow(2).sum(-1, keepdim=True) - 2 * torch.matmul(x, self.C) + self.Cnorm

        # version 2:
        # dist = - 2 * torch.matmul(x, self.C) + self.Cnorm
        return dist
        # return dist.argmin(dim=-1)
