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
The core of our approach to speech is *discretization* - continuous speech signals are converted into sequences of tokens, which can then be processed alongside text.

TODO: link to kmeans model, code snippet

## Reproducing our Results
TODO: ducttape example

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
