task LabelAudio
    > dsus=dsu.txt
    :: hubert_model=@
    :: hubert_layer=@
    :: hubert_dtype=@
    :: km_model=@
    :: dataset_config=@
    :: repo=@
{
    full_dataset_config=$dataset_config
    if [ ! -f $full_dataset_config ] ; then
        full_dataset_config="${repo}/${full_dataset_config}"
    fi

    km_model_path=$km_model
    if [ ! -f $km_model_path ] ; then
        km_model_path="${repo}/${km_model_path}"
    fi

    python $repo/scripts/label-audio.py \
        --config $full_dataset_config \
        --out-path $dsus \
        --ssl-model $hubert_model \
        --layer $hubert_layer \
        --kmeans-model $km_model_path \
        --batch-size 4 \
        --dtype $hubert_dtype
}

task BuildInstructions
    < test_dsus=$dsus@LabelAudio
    > instructions=instructions.json
    :: speech2text_tokenizer=@
    :: template=@
    :: repo=@
{
    full_template=$template
    if [ ! -f $template ] ; then
        full_template="${repo}/${full_template}"
    fi

    # template, language pair
    python $repo/scripts/build_instructions.py \
        --src $test_dsus \
        --src-names dsu_seq \
        --templates $full_template \
        --template-key asr \
        --tokenizer $speech2text_tokenizer \
        --out $instructions
}

task Decode
    < instructions=@BuildInstructions
    > hyps=hyps.txt
    :: speech2text_model=@
    :: speech2text_tokenizer=@
    :: max_tokens=@
    :: repo=@
{
    python $repo/scripts/inference.py \
        --inpaths $instructions \
        --outpaths $hyps \
        --model $speech2text_model \
        --tokenizer $speech2text_tokenizer \
        --max-length $max_tokens
}

task DumpASRText
    > references=references.txt
    > metadata=metadata.json
    :: dataset_config=@
    :: transcript_field=@
    :: repo=@
{
    full_dataset_config=$dataset_config
    if [ ! -f $template ] ; then
        full_dataset_config="${repo}/${full_dataset_config}"
    fi
    python $repo/scripts/extract_asr_text.py \
        --config $full_dataset_config \
        --text-field $transcript_field \
        --corpus $references --metadata $metadata
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
    reach LabelAudio via (ASRDataset: *) * (KMModel: *)
}

plan LabelizeBF16 {
    reach LabelAudio via (ASRDataset: *) * (HubertDtype: bf16) * (KMModel: *)
}

##### Hypothesis Generation #####

plan DecodeAll {
    reach Decode via (ASRDataset: *) * (Speech2TextModel: *) * (HubertDtype: bf16) * (KMModel: *) * (Template: *)
}

##### Evaluation #####

plan ComputeWERAll {
    reach ComputeWER via (ASRDataset: *) * (Speech2TextModel: *) * (HubertDtype: bf16) * (KMModel: *) * (Template: *)
}
