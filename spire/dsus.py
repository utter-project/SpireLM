from tqdm import tqdm

from spire.kmeans import KmeansForInference
from spire.hubert_features import HFHubertFeatureReader
from spire.data import build_dataloader
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
        loader, n_batches = build_dataloader(
            path=tsv_path,
            sample_rate=sample_rate,
            batch_size=batch_size,
            feature_extractor=self.hubert.fe,
            num_workers=num_workers
        )

        labels = []
        indices = []
        for batch in tqdm(loader, total=n_batches):
            inp = batch.input_values.cuda()
            mask = batch.attention_mask.cuda()
            batch_labels = self.label(batch=inp, indices_only=indices_only, attention_mask=mask)
            # total_tokens = inp.numel()
            # nonpad = mask.sum().item()
            # print(inp.shape, nonpad / total_tokens)
            labels.extend(batch_labels)
            indices.extend(batch.indices)
        labels = [label for i, label in sorted(zip(indices, labels))]  # return to corpus order
        return labels
