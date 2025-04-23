"""
Adapted from https://github.com/facebookresearch/fairseq/blob/main/examples/hubert/simple_kmeans/dump_hubert_feature.py,
copyright (c) Facebook, Inc. and its affiliates.
"""

import torch
import fairseq
from spire.fairseq_audio_utils import read_legacy_audio

import soundfile as sf
from transformers import Wav2Vec2FeatureExtractor


class FairseqHubertFeatureReader:
    def __init__(self, ckpt_path, layer, max_chunk=1600000, sample_rate=16000, normalize=True, legacy_audio=False):
        model, _, _, = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
        self.model = model[0].eval().cuda()
        self.layer = layer
        self.max_chunk = max_chunk
        self.sample_rate = sample_rate
        self.normalize = normalize

        self.fe = Wav2Vec2FeatureExtractor()
        self.legacy_audio = legacy_audio

    def read_audio(self, path, ref_len=None):
        if self.legacy_audio:
            return read_legacy_audio(path, sample_rate=self.sample_rate, normalize=self.normalize)
        wav, sample_rate = sf.read(path)  # second value is sample rate
        assert sample_rate == self.sample_rate, "Expected sample rate {}, got {}".format(self.sample_rate, sample_rate)

        features = self.fe(wav, sampling_rate=self.sample_rate, return_tensors="pt").input_values

        assert wav.shape[0] == features.shape[1], (wav.shape, features.shape)
        return features.cuda()

    def get_feats(self, path, ref_len=None):
        with torch.no_grad():
            x = self.read_audio(path, ref_len=ref_len)

            feat = []
            for start in range(0, x.size(1), self.max_chunk):
                x_chunk = x[:, start: start + self.max_chunk]
                feat_chunk, _ = self.model.extract_features(
                    source=x_chunk,
                    padding_mask=None,
                    mask=False,
                    output_layer=self.layer,
                )
                feat.append(feat_chunk)
        return torch.cat(feat, 1).squeeze(0)
