from tqdm import tqdm

import torch

from spire.kmeans import KmeansForInference
from spire.hubert_features import HFHubertFeatureReader
from spire.utils import indices2dsus, deduplicate
from spire.hubert_labeler import HubertLabeler


# this whole thing needs to be refactored, I think. And that's fine.
# but also, the feature-reading/kmeans could be refactored into a single
# nn.Module, I think. Pass ckpt_path, layer, and kmeans path. Forward generates
# scores for each cluster, and you implement a predict method as well that picks
# the closest cluster.
class Labeler:

    def __init__(
        self, ckpt_path, km_path, feature_layer=22, deduplicated=True, dtype=torch.float32
    ):

        self.model = HubertLabeler(ckpt_path, km_path, layer=feature_layer, dtype=dtype).eval().cuda()
        self.deduplicated = deduplicated

    def label(self, path=None, batch=None, indices_only=False, attention_mask=None):


        # feats = self.hubert.get_feats(path=path, batch=batch, attention_mask=attention_mask)  # b x len x dim
        # labels = self.kmeans(feats)

        # huh, the masking should probably happen inside the labeler
        labels = self.model.predict(batch)

        # This block is detokenization: account for deduplication and detensorizing
        # the predictions
        labels = labels.to("cpu").tolist()  # should be a list of lists
        if self.deduplicated:
            labels = [deduplicate([l for l in lab if l != -1]) for lab in labels]

        if indices_only:
            return labels
        labels = [indices2dsus(ix) for ix in labels]
        return labels
