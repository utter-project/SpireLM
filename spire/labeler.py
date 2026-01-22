import torch
import torch.nn as nn
import joblib
from transformers import AutoModel, HubertModel, Wav2Vec2BertModel

from spire.utils import load_wav, detokenize


def _pool_out_length(input_length, pooling_size, stride=None):
    # 1D convolutional layer output length formula taken
    # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
    if stride is None:
        stride = pooling_size
    return (torch.div(input_length - pooling_size, stride, rounding_mode="floor") + 1).to(torch.long)


def _lengths_to_mask(lengths, max_length, dtype, device):
    batch_size = lengths.shape[0]
    attention_mask = torch.zeros(
        (batch_size, max_length), dtype=dtype, device=device
    )
    attention_mask[(torch.arange(attention_mask.shape[0], device=device), lengths - 1)] = 1
    attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).bool()
    return attention_mask


class Featurizer(nn.Module):

    def __init__(self, ckpt_path, layer=22, dtype=torch.float32, pooling_width=1, pooling_type="mean"):
        super().__init__()

        self.model = AutoModel.from_pretrained(ckpt_path, torch_dtype=dtype)

        if isinstance(self.model, HubertModel):
            self.model.encoder.layer_norm = nn.Identity()  # kludge to avoid applying final layer norm
            self.model.encoder.layers = self.model.encoder.layers[:layer]
        elif isinstance(self.model, Wav2Vec2BertModel):
            # I *think* this is the way to go for getting SSL features from a
            # w2v model
            self.model.encoder.layers = self.model.encoder.layers[:layer]
            self.model.encoder.layers[-1].final_layer_norm = nn.Identity()
        else:
            raise ValueError("Unknown SSL architecture")

        if pooling_width > 1:
            # do I want ceil_mode?
            if pooling_type == "mean":
                self.pooling = nn.AvgPool1d(pooling_width, ceil_mode=True)
            else:
                self.pooling = nn.MaxPool1d(pooling_width, ceil_mode=True)
        else:
            self.pooling = None

    def _get_feature_vector_attention_mask(self, length, attention_mask):
        return self.model._get_feature_vector_attention_mask(length, attention_mask)

    def forward(self, batch, attention_mask=None, flatten=False, return_pad_percent=False):
        """
        Take a batch of inputs, return scores for all V clusters
        If flatten == False, return batch x seq_len x D
        If flatten == True, return N x D, where N <= batch*seq_len == the number of non-pad positions
        """
        feats = self.model(batch, attention_mask=attention_mask).last_hidden_state
        pre_pool_max_len = feats.shape[1]

        if self.pooling is not None:
            feats = self.pooling(feats.transpose(1, 2)).transpose(1, 2)
        post_pool_max_len = feats.shape[1]

        if not flatten:
            return feats

        assert attention_mask is not None

        # compute pre-pool output mask
        output_mask = self._get_feature_vector_attention_mask(
            pre_pool_max_len,
            attention_mask
        )

        if self.pooling is not None:
            post_pool_lengths = _pool_out_length(
                output_mask.sum(dim=-1),
                self.pooling.kernel_size[0],
                self.pooling.stride[0]
            )
            # convert length to mask
            output_mask = _lengths_to_mask(
                post_pool_lengths,
                post_pool_max_len,
                output_mask.dtype,
                output_mask.device
            )

        flattened_output_mask = output_mask.view(-1)

        flattened_feats = feats.view(-1, feats.shape[-1])
        non_pad_feats = flattened_feats[flattened_output_mask]

        if not return_pad_percent:
            return non_pad_feats

        n_before_flat = flattened_feats.shape[0]
        n_after_flat = non_pad_feats.shape[0]

        return non_pad_feats, 1 - n_after_flat / n_before_flat


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


class Labeler(nn.Module):

    def __init__(self, ckpt_path, km_path, layer=22, dtype=torch.float32, pooling_width=1, pooling_type="mean"):
        """
        The layer default of 22 is a strong value for HuBERT-large. It may be
        inappropriate for other models.
        """
        super().__init__()

        self.featurizer = Featurizer(
            ckpt_path,
            layer=layer,
            dtype=dtype,
            pooling_width=pooling_width,
            pooling_type=pooling_type
        )
        self.kmeans = KMeans(km_path, dtype=dtype)

    def forward(self, batch, attention_mask=None):
        """
        Take a batch of inputs, return scores for all V clusters
        """

        feats = self.featurizer(batch, attention_mask=attention_mask)
        dist = self.kmeans(feats)
        return dist

    def predict(self, batch, attention_mask=None):
        attention_mask = None
        dist = self(batch, attention_mask=attention_mask)
        labels = dist.argmin(dim=-1)

        # what's feats.shape[1]? It's the length dimension, so it should be
        # dist.shape[1] as well
        if attention_mask is not None:
            # right ... if we pool, then dist.shape[1] doesn't match attention_mask and it fails.
            output_mask = self._get_feature_vector_attention_mask(dist.shape[1], attention_mask)
            print("output_mask", output_mask.sum(dim=-1))
            labels.masked_fill_(~output_mask, -1)
        return labels

    def _get_feature_vector_attention_mask(self, length, attention_mask):
        return self.featurizer._get_feature_vector_attention_mask(length, attention_mask)

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
