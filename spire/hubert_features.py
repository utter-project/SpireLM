"""
Adapted from https://github.com/facebookresearch/fairseq/blob/main/examples/hubert/simple_kmeans/dump_hubert_feature.py,
copyright (c) Facebook, Inc. and its affiliates.
"""

from os.path import join

import torch

import soundfile as sf
from transformers import HubertModel, Wav2Vec2FeatureExtractor


class HFHubertFeatureReader:

    def __init__(self, ckpt_path, layer, sample_rate=16000):
        self.model = HubertModel.from_pretrained(ckpt_path).eval().cuda()
        self.model.encoder.layer_norm = torch.nn.Identity()  # kludge to avoid applying final layer norm

        self.model.encoder.layers = self.model.encoder.layers[:layer]  # not an obob


        self.fe = Wav2Vec2FeatureExtractor()
        self.sample_rate = sample_rate

    def get_feats(self, batch=None, path=None, ref_len=None, attention_mask=None):
        assert (batch is not None) != (path is not None), \
            "Must specify batch or path but not both"
        with torch.no_grad():
            if path is not None:
                # read the path, preprocess it
                wav, sample_rate = sf.read(path, dtype="float32")
                assert sample_rate == self.sample_rate, "Expected sample rate {}, got {}".format(self.sample_rate, sample_rate)

                # do we need to do any preprocessing here? I don't think so because
                # HuBERT forward already calls the FeatureExtractor.
                # it seems wrong to do the feature extraction here, given that it also happens
                # inside the model forward call, but it produces better ASR results for SpireFull
                batch = self.fe(wav, sampling_rate=self.sample_rate, return_tensors="pt").input_values.cuda()

            # return batch x len x dim
            feats = self.model(batch, attention_mask=attention_mask).last_hidden_state
            return feats
