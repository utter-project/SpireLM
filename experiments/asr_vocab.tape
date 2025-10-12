task CreateReducedVocab
    > vocab=vocab.txt
    :: corpus_freqs=@
    :: repo=@
    :: vocab_size=@
    :: metric=@
    :: km_model=@
{
    python $repo/scripts/create_reduced_vocab.py \
        --freqs $corpus_freqs \
        --metric $metric \
        --vocab-size $vocab_size \
        --kmeans $km_model > $vocab
}

task LabelAudio
    < vocab=@CreateReducedVocab
    > dsus=dsu.txt
    :: hubert_model=@
    :: hubert_layer=@
    :: km_model=@
    :: hubert_dtype=@
    :: asr_dataset=@
    :: asr_path_extra=@
    :: repo=@
{
    # dataset_path should be defined in the uservars file
    # dataset_dype should be hf-disk
    # resample-to should not be necessary
    # batch-size should be 4 (since it isn't token-batching anymore)
    python $repo/scripts/label-audio.py \
        --tsv_path $asr_dataset \
        --dataset-type "hf-disk" \
        --path-extra $asr_path_extra \
        --hf-split test \
        --out_path $dsus \
        --ckpt_path $hubert_model \
        --layer $hubert_layer \
        --km_path $km_model \
        --batch-size 4 \
        --dtype $hubert_dtype \
        --vocabulary $vocab
}

task BuildInstructions
    < test_dsus=$dsus@LabelAudio
    > instructions=instructions.json
    :: template=@
    :: speech2text_tokenizer=@
    :: repo=@
{
    # template, language pair, number of shots
    python $repo/scripts/build_instructions.py \
        --src $test_dsus \
        --template $template \
        --n-shots 0 \
        --chat-tokenizer $speech2text_tokenizer \
        --out $instructions
}

task Decode
    < instructions=@BuildInstructions
    > hyps=hyps.txt
    :: speech2text_model=@
    :: speech2text_tokenizer=@
    :: max_tokens=@
    :: backend=@
    :: repo=@
{
    python $repo/scripts/inference.py \
        --inpaths $instructions \
        --outpaths $hyps \
        --model $speech2text_model \
        --tokenizer $speech2text_tokenizer \
        --backend $backend \
        --max-length $max_tokens

}

task DumpASRText
    > references=references.txt
    > metadata=metadata.json
    :: asr_dataset=@
    :: asr_path_extra=@
    :: asr_split=@
    :: asr_text_field=@
    :: repo=@
{
    python $repo/scripts/extract_asr_text.py \
        --path $asr_dataset --path-extra $asr_path_extra --split $asr_split --text-field $asr_text_field \
        --dataset-type hf-disk --corpus $references --metadata $metadata
}

task NormalizeHyps
    < hyps=@Decode
    > hyps_normalized=hyps.txt.norm
    :: normalizer=@
    :: repo=@
{
    python $repo/scripts/normalize.py --normalizer $normalizer < $hyps > $hyps_normalized
}

task NormalizeRefs
    < references=@DumpASRText
    > refs_normalized=refs.txt.norm
    :: normalizer=@
    :: repo=@
{
    python $repo/scripts/normalize.py --normalizer $normalizer < $references > $refs_normalized
}

task ComputeWER
    < hyps_normalized=@NormalizeHyps
    < refs_normalized=@NormalizeRefs
    > results=wer.json
    :: repo=@
{
    python $repo/scripts/compute_wer.py --hyp $hyps_normalized --ref $refs_normalized > $results
}

##### Data Preparation #####


plan Labelize {
    reach LabelAudio via (ASRDataset: fleurs) * (VocabSize: *) * (Metric: *) * (CorpusFreqs: *)
}

##### Hypothesis Generation #####

plan DecodeAll {
    reach Decode via (ASRDataset: fleurs) * (Speech2TextModel: *) * (VocabSize: *) * (Metric: *) * (CorpusFreqs: *)
}

##### Evaluation #####

plan ComputeWERAll {
    reach ComputeWER via (ASRDataset: fleurs) * (Speech2TextModel: *) * (VocabSize: *) * (Metric: *) * (CorpusFreqs: *)
}
