# SpireLM
Spire is a 7B parameter decoder-only model with strong abilities in machine translation, automatic speech recognition, and speech translation. [SpireBase](https://huggingface.co/utter-project/SpireBase) was created by applying speech-centric continued pretraining to [TowerBase-7B-v0.1](https://huggingface.co/Unbabel/TowerBase-7B-v0.1), which was itself created by applying continued pretraining to [Llama 2](https://huggingface.co/meta-llama/Llama-2-7b).

## Model Checkpoints
We release our checkpoints through Hugging Face. All of our models can be loaded as `LlamaForCausalLM` instances, allowing inference to be performed with [vLLM](https://github.com/vllm-project/vllm). For further details on the models, check [the paper](https://arxiv.org/abs/2503.10620).

| Model | Path |
| ----- | ---- |
| SpireBase | [utter-project/SpireBase](https://huggingface.co/utter-project/SpireBase) |
| SpireFull | [utter-project/SpireFull](https://huggingface.co/utter-project/SpireFull) |
| SpireNoBlocks | [utter-project/SpireNoBlocks](https://huggingface.co/utter-project/SpireNoBlocks) |
| SpireNoPseudo | [utter-project/SpireNoBlocks](https://huggingface.co/utter-project/SpireNoPseudo) |
| TowerFull | [utter-project/TowerFull](https://huggingface.co/utter-project/TowerFull) |

## Tokenizing Speech
The core of our approach to speech is *discretization* - continuous speech signals are converted into sequences of tokens, which can then be processed alongside text. Our discretization system consists of a few steps:

1. HuBERT Large ([fairseq download](https://dl.fbaipublicfiles.com/hubert/hubert_large_ll60k.pt)) converts 16kHz .wav files into into a sequence of feature vectors, one for each 20ms frame. We use the representations from layer 22.
2. Our k-means model ([download](https://huggingface.co/utter-project/SpireKMeans/resolve/main/kmeans_model)) maps each frame to one of 5000 clusters.
3. The sequences of cluster IDs are deduplicated, such that consecutive frames with the same label are collapsed into a single token. This usually shortens the sequence length by about 30%.

The `spire` package implements this pipeline. Assuming you have downloaded both of these files, you can use it like so:

```
from datasets import load_dataset
from spire.dsus import Labeler
from spire.utils import fix_fleurs_path

fleurs = load_dataset("google/fleurs", "en_us")
wav = fix_fleurs_path(fleurs["validation"][29], "validation")

labeler = Labeler("hubert_large_ll60k.pt", "kmeans_model")
speech_tokens = labeler.label(wav)
print(speech_tokens)
```

The output will not be very readable, as it consists of a sequence of Unicode [private use area](https://en.wikipedia.org/wiki/Private_Use_Areas) characters. However, these characters are known to the Spire tokenizer and can be combined with text:

TODO: add ASR/ST examples with this sequence

## Reproducing our Inference Results
TODO: ducttape example

## Reproducing our Training

## Citation
If you use Spire, please cite our work:
```
@misc{spire,
      title={From TOWER to SPIRE: Adding the Speech Modality to a Text-Only LLM}, 
      author={Kshitij Ambilduke and Ben Peters and Sonal Sannigrahi and Anil Keshwani and Tsz Kin Lam and Bruno Martins and Marcely Zanon Boito and André F. T. Martins},
      year={2025},
      eprint={2503.10620},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2503.10620}
}
```
