"""
Adapted from https://github.com/facebookresearch/fairseq/blob/main/examples/hubert/simple_kmeans/dump_hubert_feature.py,
copyright (c) Facebook, Inc. and its affiliates.
"""

import torch
import torch.nn.functional as F
import fairseq
from spire.fairseq_audio_utils import get_features_or_waveform


class FairseqHubertFeatureReader:
    def __init__(self, ckpt_path, layer, max_chunk=1600000, sample_rate=16000, normalize=True):
        model, _, _, = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
        self.model = model[0].eval().cuda()
        self.layer = layer
        self.max_chunk = max_chunk
        self.sample_rate = sample_rate
        self.normalize = normalize

    def read_audio(self, path, ref_len=None):
        wav = get_features_or_waveform(path, use_sample_rate=self.sample_rate)

        if wav.ndim == 2:
            wav = wav.mean(-1)

        assert wav.ndim == 1, wav.ndim
        '''
        if ref_len is not None and abs(ref_len - len(wav)) > 160:
            logging.warning(f"ref {ref_len} != read {len(wav)} ({path})")
        '''
        return wav

    def get_feats(self, path, ref_len=None):
        x = self.read_audio(path, ref_len=ref_len)
        with torch.no_grad():
            x = torch.from_numpy(x).float().cuda()

            if self.normalize:
                x = F.layer_norm(x, x.shape)
            x = x.view(1, -1)

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
