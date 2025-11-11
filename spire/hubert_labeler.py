import torch
import torch.nn as nn
import joblib
from transformers import HubertModel

from spire.utils import load_wav, detokenize


class Featurizer(nn.Module):

    def __init__(self, ckpt_path, layer=22, dtype=torch.float32):
        super().__init__()

        self.model = HubertModel.from_pretrained(ckpt_path, torch_dtype=dtype)
        self.model.encoder.layer_norm = nn.Identity()  # kludge to avoid applying final layer norm
        self.model.encoder.layers = self.model.encoder.layers[:layer]

    def forward(self, batch, attention_mask=None):
        """
        Take a batch of inputs, return scores for all V clusters
        """
        feats = self.model(batch, attention_mask=attention_mask).last_hidden_state
        return feats


class KMeans(nn.Module):
    """
    Adapted from https://github.com/facebookresearch/fairseq/blob/main/examples/hubert/simple_kmeans/dump_km_label.py,
    copyright (c) Facebook, Inc. and its affiliates.

    K-means adapted from fairseq. Note that this may produce slightly different
    cluster assignments than the original numpy-based implementation due to
    numerical differences. Likewise, changing the dtype can yield marginally
    different clusters, but empirically this doesn't seem to matter much.
    """
    def __init__(self, km_path, dtype=torch.float32):
        super().__init__()

        km_model = joblib.load(km_path)
        C = km_model.cluster_centers_.transpose()
        Cnorm = (C ** 2).sum(0, keepdims=True)

        self.C = nn.Parameter(torch.from_numpy(C).to(dtype))
        self.Cnorm = nn.Parameter(torch.from_numpy(Cnorm).to(dtype))

    def forward(self, x):
        """
        x: * x dim
        """
        # self.Cnorm is essentially a bias...
        dist = x.pow(2).sum(-1, keepdim=True) - 2 * torch.matmul(x, self.C) + self.Cnorm

        # version 2:
        # dist = - 2 * torch.matmul(x, self.C) + self.Cnorm
        return dist


class HubertLabeler(nn.Module):

    def __init__(self, ckpt_path, km_path, layer=22, dtype=torch.float32):
        super().__init__()

        self.featurizer = Featurizer(ckpt_path, layer=layer, dtype=dtype)
        self.kmeans = KMeans(km_path, dtype=dtype)

    def forward(self, batch, attention_mask=None):
        """
        Take a batch of inputs, return scores for all V clusters
        """

        feats = self.featurizer(batch, attention_mask=attention_mask)
        dist = self.kmeans(feats)
        return dist

    def predict(self, batch, attention_mask=None):
        dist = self(batch, attention_mask=attention_mask)
        labels = dist.argmin(dim=-1)

        # what's feats.shape[1]? It's the length dimension, so it should be
        # dist.shape[1] as well
        if attention_mask is not None:
            output_mask = self._get_feature_vector_attention_mask(dist.shape[1], attention_mask)
            labels.masked_fill_(~output_mask, -1)
        return labels

    def _get_feature_vector_attention_mask(self, length, attention_mask):
        return self.featurizer.model._get_feature_vector_attention_mask(length, attention_mask)

    def label_wav(self, wav_path, **detok_args):
        # read the audio into a batch
        device = self.kmeans.C.device
        batch = load_wav(
            wav_path, device=device, expected_sample_rate=16000
        )

        # no attention mask because it's a single-element batch
        labels = self.predict(batch)

        # detokenize
        detokenized_labels = detokenize(labels, **detok_args)

        return detokenized_labels


class DPDPHubertLabeler(nn.Module):

    def __init__(self, ckpt_path, km_path, layer=22, dtype=torch.float32, lmbda=0.):
        super().__init__()
        # featurizer is the same as for vanilla Hubert labels.
        self.featurizer = Featurizer(ckpt_path, layer=layer, dtype=dtype)

        # the quantizer wants it [T x D], whereas KMeans.C is [D x T]
        codebook = KMeans(km_path, dtype=dtype).C.T

        self.quantizer = torch.hub.load(
            "nicolvisser/dpdp",
            "dpdp_quantizer_from_codebook",
            codebook=codebook,
            lmbda=lmbda, # <- control coarseness here
            num_neighbors=None,  # not sure if this makes a difference here
            trust_repo=True,
            force_reload=False
        )

    # there will probably be room for some inheritance
    def forward(self, batch, attention_mask=None):
        """
        Take a batch of inputs, return scores for all V clusters
        """

        feats = self.featurizer(batch, attention_mask=attention_mask)
        # the quantizer does not support batching, so we do this in a comprehension
        predictions = torch.stack(
            [self.quantizer(feats_i)[1] for feats_i in feats]
        )
        return predictions

    def predict(self, batch, attention_mask=None):
        labels = self(batch, attention_mask=attention_mask)

        if attention_mask is not None:
            output_mask = self._get_feature_vector_attention_mask(labels.shape[1], attention_mask)
            labels.masked_fill_(~output_mask, -1)
        return labels

    def _get_feature_vector_attention_mask(self, length, attention_mask):
        return self.featurizer.model._get_feature_vector_attention_mask(length, attention_mask)
