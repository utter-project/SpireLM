import torch
import torch.nn as nn
import joblib
from transformers import AutoModel, AutoConfig
from transformers.models.whisper.modeling_whisper import WhisperEncoder


def adapt_hubert(model, layer=None, remove_final_layer_norm=True):
    if remove_final_layer_norm:
        model.encoder.layer_norm = nn.Identity()  # kludge to avoid applying final layer norm
    if layer is not None:
        model.encoder.layers = model.encoder.layers[:layer]
    return model


def adapt_w2v_bert(model, layer=None, remove_final_layer_norm=True):
    if layer is not None:
        model.encoder.layers = model.encoder.layers[:layer]
    if remove_final_layer_norm:
        model.encoder.layers[-1].final_layer_norm = nn.Identity()
    return model


def adapt_whisper(model, layer=None, remove_final_layer_norm=True):
    model = model.encoder
    if layer is not None:
        model.layers = model.layers[:layer]
    if remove_final_layer_norm:
        model.layers[-1].final_layer_norm = nn.Identity()
    return model


# keys are with AutoConfig.from_pretrained(path).model_type
MODEL_ADAPTERS = {
    "hubert": adapt_hubert,
    "wav2vec2-bert": adapt_w2v_bert,
    "whisper": adapt_whisper,
    "wav2vec2": adapt_hubert,
}


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

    def __init__(self, ckpt_path, layer=None, dtype=torch.float32, pooling_width=1, pooling_type="mean", keep_final_layer_norm=False):
        super().__init__()

        self.model = self._init_model(ckpt_path, layer, dtype, keep_final_layer_norm)

        if pooling_width > 1:
            # do I want ceil_mode?
            if pooling_type == "mean":
                self.pooling = nn.AvgPool1d(pooling_width, ceil_mode=False)
            else:
                self.pooling = nn.MaxPool1d(pooling_width, ceil_mode=False)
        else:
            self.pooling = None

    def _init_model(self, ckpt_path, layer, dtype, keep_final_layer_norm):
        model_type = AutoConfig.from_pretrained(ckpt_path).model_type
        adapter = MODEL_ADAPTERS.get(model_type, None)
        if adapter is None:
            raise ValueError(f"Unknown model type {model_type}")
        model = AutoModel.from_pretrained(ckpt_path, torch_dtype=dtype)
        model = adapter(model, layer=layer, remove_final_layer_norm=not keep_final_layer_norm)
        return model

    def _get_feature_vector_attention_mask(self, length, attention_mask):
        """
        Compute the attention mask for the output of the featurizer. For models like HuBERT, this
        means calling their internal _get_feature_vector_attention_mask. For Whisper, no such function
        exists, so it needs to be downsampled here. The input attention_mask will be of shape batch x 3000.
        The output will be downsampled to batch x 1500.
        """
        if isinstance(self.model, WhisperEncoder):
            # downsample by a factor of 2, taking the max of each pair of adjacent positions
            attention_mask = attention_mask.unsqueeze(1)  # batch x 1 x seq_len
            attention_mask = torch.nn.functional.max_pool1d(attention_mask.float(), kernel_size=2, stride=2)
            return attention_mask.squeeze(1).bool()  # batch x seq_len // 2
        else:
            return self.model._get_feature_vector_attention_mask(length, attention_mask)

    def forward(self, batch, attention_mask=None, flatten=False, return_pad_percent=False):
        """
        Take a batch of inputs, return scores for all V clusters
        If flatten == False, return batch x seq_len x D
        If flatten == True, return N x D, where N <= batch*seq_len == the number of non-pad positions
        """
        assert attention_mask is not None

        feats = self.model(batch, attention_mask=attention_mask).last_hidden_state
        pre_pool_max_len = feats.shape[1]  # always 1500 for whisper models

        if self.pooling is not None:
            feats = self.pooling(feats.transpose(1, 2)).transpose(1, 2)
        post_pool_max_len = feats.shape[1]

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

        if not flatten:
            return feats, output_mask

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

    def __init__(self, ckpt_path, km_path, layer=None, dtype=torch.float32, pooling_width=1, pooling_type="mean", keep_final_layer_norm=False):
        """
        22 is a strong default value for HuBERT-large. It may be
        inappropriate for other models.
        """
        super().__init__()

        self.featurizer = Featurizer(
            ckpt_path,
            layer=layer,
            dtype=dtype,
            pooling_width=pooling_width,
            pooling_type=pooling_type,
            keep_final_layer_norm=keep_final_layer_norm
        )
        self.kmeans = KMeans(km_path, dtype=dtype)

    def forward(self, batch, attention_mask=None):
        """
        Take a batch of inputs, return scores for all V clusters
        """

        feats, attention_mask = self.featurizer(batch, attention_mask=attention_mask)
        dist = self.kmeans(feats)
        return dist, attention_mask

    def predict(self, batch, attention_mask=None):
        dist, output_mask = self(batch, attention_mask=attention_mask)
        labels = dist.argmin(dim=-1)

        if output_mask is not None:
            labels.masked_fill_(~output_mask, -1)
        return labels
