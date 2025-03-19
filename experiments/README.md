# Reproducing SpireLM Results

## ducttape
Our experiment results depend on [ducttape](https://github.com/coderpat/ducttape), whose installation instructions are in the linked repository.
ducttape allows our experiments to be decomposed into a dependency graph.
Calling ducttape with a specific plan (indicated by the `-p` flag) will execute not only the task(s) explicitly identified by the plan, but also their dependencies.
So for example, a plan that computes the WER of a model on a specific dataset may need to download the dataset, convert it to the correct format, convert the audio to discrete tokens, generate the predictions, and then compute the WER.
Although it is often possible to run the entire pipeline using a single plan, this is not always desirable: in the example above, discretizing audio and running the model require a GPU, but the other tasks do not, so it makes sense to run the GPU tasks separately from the CPU ones.
ducttape will remember which tasks it has already run and will store their outputs.
Separating the plans is also a good idea if different tasks require different environments.

## Setting up uservars files
TODO

## ASR Results

### Compute ASR results for all SpireLM models

```
ducttape asr.tape -C tape-configs/asr.tconf -p Labelize
ducttape asr.tape -C tape-configs/asr.tconf -p DecodeTowerSpire
ducttape asr.tape -C tape-configs/asr.tconf -p ComputeWERTowerSpire
```

### Compute ASR results for Seamless

```
ducttape asr.tape -C tape-configs/asr.tconf -p DecodeSL
ducttape asr.tape -C tape-configs/asr.tconf -p ComputeSeamlessWER
```

### Compute ASR results for Whisper

```
ducttape asr.tape -C tape-configs/asr.tconf -p DecodeWh
ducttape asr.tape -C tape-configs/asr.tconf -p ComputeWhisperWER
```
