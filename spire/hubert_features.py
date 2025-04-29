"""
Adapted from https://github.com/facebookresearch/fairseq/blob/main/examples/hubert/simple_kmeans/dump_hubert_feature.py,
copyright (c) Facebook, Inc. and its affiliates.
"""

import torch
from spire.fairseq_audio_utils import read_legacy_audio

import soundfile as sf
from transformers import HubertModel, Wav2Vec2FeatureExtractor


class HFHubertFeatureReader:

    def __init__(self, ckpt_path, layer, sample_rate=16000):
        # easy-peasy, right?
        self.model = HubertModel.from_pretrained(ckpt_path).eval().cuda()
        self.model.encoder.layer_norm = torch.nn.Identity()  # kludge to avoid applying final layer norm

        self.model.encoder.layers = self.model.encoder.layers[:layer]  # not an obob

        # can't use the processor class because our hubert model has no tokenizer
        # self.processor = AutoProcessor.from_pretrained(ckpt_path)

        self.fe = Wav2Vec2FeatureExtractor()
        self.sample_rate = sample_rate

    def get_feats(self, path, ref_len=None):
        with torch.no_grad():
            # But first, read in the file
            wav, sample_rate = sf.read(path, dtype="float32")  # second value is sample rate
            assert sample_rate == self.sample_rate, "Expected sample rate {}, got {}".format(self.sample_rate, sample_rate)

            # do we need to do any preprocessing here? I don't think so because
            # HuBERT forward already calls the FeatureExtractor.
            # it seems wrong to do the feature extraction here, given that it also happens
            # inside the model forward call, but it produces better ASR results for SpireFull
            input_values = self.fe(wav, sampling_rate=self.sample_rate, return_tensors="pt").input_values.cuda()

            outputs = self.model(input_values)
            feats = outputs.last_hidden_state.squeeze(0)  # return len x dim
            return feats
