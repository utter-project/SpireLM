from functools import partial

from tqdm import tqdm
from torch.utils.data import DataLoader

from spire.kmeans import KmeansForInference
from spire.hubert_features import HFHubertFeatureReader
from spire.data import AudioTSVDataset, collate_fn
from spire.utils import indices2dsus, deduplicate


# this whole thing needs to be refactored, I think. And that's fine.
# but also, the feature-reading/kmeans could be refactored into a single
# nn.Module, I think. Pass ckpt_path, layer, and kmeans path. Forward generates
# scores for each cluster, and you implement a predict method as well that picks
# the closest cluster.
class Labeler:

    def __init__(
        self, ckpt_path, km_path, feature_layer=22, deduplicated=True, kmeans_device="cuda:0"
    ):
        self.hubert = HFHubertFeatureReader(ckpt_path, feature_layer)
        self.kmeans = KmeansForInference(km_path, device=kmeans_device)
        self.deduplicated = deduplicated

    def label(self, path=None, batch=None, indices_only=False, attention_mask=None):
        feats = self.hubert.get_feats(path=path, batch=batch, attention_mask=attention_mask)  # b x len x dim
        labels = self.kmeans(feats)

        if attention_mask is not None:
            output_mask = self.hubert.model._get_feature_vector_attention_mask(feats.shape[1], attention_mask)
            labels.masked_fill_(~output_mask, -1)

        # This block is detokenization: account for deduplication and detensorizing
        # the predictions
        labels = labels.to("cpu").tolist()  # should be a list of lists
        if self.deduplicated:
            labels = [deduplicate([l for l in lab if l != -1]) for lab in labels]

        if indices_only:
            return labels
        labels = [indices2dsus(ix) for ix in labels]
        return labels

    def label_corpus(self, tsv_path, sample_rate=16000, indices_only=False, batch_size=1, num_workers=0):
        dataset = AudioTSVDataset(tsv_path, sample_rate=sample_rate)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=partial(collate_fn, feature_extractor=self.hubert.fe),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False
        )

        labels = []
        for batch in tqdm(loader):
            inp = batch.input_values.cuda()
            mask = batch.attention_mask.cuda()
            batch_labels = self.label(batch=inp, indices_only=indices_only, attention_mask=mask)
            # total_tokens = inp.numel()
            # nonpad = mask.sum().item()
            labels.extend(batch_labels)
        return labels
