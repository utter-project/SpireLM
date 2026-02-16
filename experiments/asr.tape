task LabelAudio
    > dsus=dsu.txt
    :: hubert_model=@
    :: hubert_layer=@
    :: km_model=@
    :: hubert_dtype=@
    :: asr_dataset=@
    :: asr_path_extra=@
    :: dataset_type=@
    :: repo=@
{
    # dataset_path should be defined in the uservars file
    # dataset_dype should be hf-disk
    # resample-to should not be necessary
    # batch-size should be 4 (since it isn't token-batching anymore)
    python $repo/scripts/label-audio.py \
        --tsv_path $asr_dataset \
        --dataset-type $dataset_type \
        --path-extra $asr_path_extra \
        --hf-split test \
        --out_path $dsus \
        --ckpt_path $hubert_model \
        --layer $hubert_layer \
        --km_path $km_model \
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
    # template, language pair
    python $repo/scripts/build_instructions.py \
        --src $test_dsus \
        --src-names dsu_seq \
        --templates $template \
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
    :: asr_dataset=@
    :: asr_path_extra=@
    :: dataset_type=@
    :: asr_split=@
    :: asr_text_field=@
    :: repo=@
{
    python $repo/scripts/extract_asr_text.py \
        --path $asr_dataset --path-extra $asr_path_extra --split $asr_split --text-field $asr_text_field \
        --dataset-type $dataset_type --corpus $references --metadata $metadata
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

plan Convert {
    reach ConvertTSV via (ASRDataset: ls_clean ls_other)
}

plan Labelize {
    reach LabelAudio via (ASRDataset: *)
}

plan LabelizeBF16 {
    reach LabelAudio via (ASRDataset: *) * (HubertDtype: bf16)
}

##### vctk30 Experiments #####

plan DecodeVCTK30 {
    reach Decode via (ASRDataset: *) * (Speech2TextModel: lmt_4B_vctk30h_400 lmt_4B_vctk30h_3600) * (KMModel: vctk_mic1_30h) * (HubertDtype: bf16)
}

plan ComputeWERVCTK30 {
    reach ComputeWER via (ASRDataset: *) * (Speech2TextModel: lmt_4B_vctk30h_400 lmt_4B_vctk30h_3600) * (KMModel: vctk_mic1_30h) * (HubertDtype: bf16)
}

##### giga30 Experiments #####

plan DecodeGiga30 {
    reach Decode via (ASRDataset: *) * (Speech2TextModel: lmt_4B_giga30h_1200 lmt_4B_giga30h_2400 lmt_4B_giga30h) * (KMModel: giga30h) * (HubertDtype: bf16)
}

plan ComputeWERGiga30 {
    reach ComputeWER via (ASRDataset: *) * (Speech2TextModel: lmt_4B_giga30h_1200 lmt_4B_giga30h_2400 lmt_4B_giga30h) * (KMModel: giga30h) * (HubertDtype: bf16)
}

##### mh-CV20 Experiments #####

plan DecodeMHCV20 {
    reach Decode via (ASRDataset: *) * (Speech2TextModel: lmt_4B_mh_cv20h_2000 lmt_4B_mh_cv20h) * (KMModel: mhubert_25hz_cv20h) * (HubertDtype: bf16)
}

plan ComputeWERMHCV20 {
    reach ComputeWER via (ASRDataset: *) * (Speech2TextModel: lmt_4B_mh_cv20h_2000 lmt_4B_mh_cv20h) * (KMModel: mhubert_25hz_cv20h) * (HubertDtype: bf16)
}

##### giga20 Experiments #####

plan DecodeGiga20 {
    reach Decode via (ASRDataset: *) * (Speech2TextModel: lmt_4B_giga20h_2000 lmt_4B_giga20h_2400 lmt_4B_giga20h_4000 lmt_4B_giga20h_4400 lmt_4B_giga20h_5600 lmt_4B_giga20h) * (KMModel: giga20h) * (HubertDtype: bf16)
}

plan ComputeWERGiga20 {
    reach ComputeWER via (ASRDataset: *) * (Speech2TextModel: lmt_4B_giga20h_2000 lmt_4B_giga20h_2400 lmt_4B_giga20h_4000 lmt_4B_giga20h_4400 lmt_4B_giga20h_5600 lmt_4B_giga20h) * (KMModel: giga20h) * (HubertDtype: bf16)
}

##### CV20 Experiments #####

plan DecodeCV20 {
    reach Decode via (ASRDataset: *) * (Speech2TextModel: lmt_4B_cv20h_1200 lmt_4B_cv20h_2000 lmt_4B_cv20h_3600 lmt_4B_cv20h_5600 lmt_4B_cv20h) * (KMModel: cv20h) * (HubertDtype: bf16)
}

plan ComputeWERCV20 {
    reach ComputeWER via (ASRDataset: *) * (Speech2TextModel: lmt_4B_cv20h_1200 lmt_4B_cv20h_2000 lmt_4B_cv20h_3600 lmt_4B_cv20h_5600 lmt_4B_cv20h) * (KMModel: cv20h) * (HubertDtype: bf16)
}

##### Hypothesis Generation #####

plan DecodeAll {
    reach Decode via (ASRDataset: *) * (Speech2TextModel: *)
}

plan DecodeSpireFull {
    reach Decode via (ASRDataset: *) * (Speech2TextModel: spire_full)
}

plan DecodeSpireFullBF16 {
    reach Decode via (ASRDataset: *) * (Speech2TextModel: spire_full) * (HubertDtype: bf16)
}

plan DecodeTowerSpire {
    reach Decode via (ASRDataset: *) * (Speech2TextModel: spire_base spire_full spire_no_pseudo spire_no_blocks tower_full)
}

##### Evaluation #####

plan ComputeWERAll {
    reach ComputeWER via (ASRDataset: *) * (Speech2TextModel: *)
}

plan ComputeWERSpireFull {
    reach ComputeWER via (ASRDataset: *) * (Speech2TextModel: spire_full)
}

plan ComputeWERSpireFullBF16 {
    reach ComputeWER via (ASRDataset: *) * (Speech2TextModel: spire_full) * (HubertDtype: bf16)
}

plan ComputeWERTowerSpire {
    reach ComputeWER via (ASRDataset: *) * (Speech2TextModel: spire_base spire_full spire_no_pseudo spire_no_blocks tower_full)
}
