# the BuildNativeTSV and ConvertTSV tasks will become obsolete, I think
task BuildNativeTSV
    :: asr_dataset=@
    :: asr_path_extra=@
    :: asr_split=@
    :: repo=@
    > tsv=audio.tsv
{
    # build a tsv file for an ASR dataset
    python $repo/scripts/build_tsv.py --path $asr_dataset --path-extra $asr_path_extra --split $asr_split --out $tsv
}

task ConvertTSV
    < native_tsv=$tsv@BuildNativeTSV
    > wav_tsv=audio.tsv
    > wav
    :: hubert_model=@
    :: hubert_layer=@
    :: repo=@
{
    mkdir -p $wav
    # convert data to wav for audio datasets in other formats (for ASR testing,
    # this should only be LibriSpeech so far)

    python $repo/scripts/convert_tsv.py --input-tsv $native_tsv --output-tsv $wav_tsv --output-dir $wav
}

# this one stays as is
task LabelVCTK
    > dsus=dsus_no_dedup.txt
    :: hubert_model=@
    :: hubert_layer=@
    :: km_model=@
    :: hubert_dtype=@
    :: repo=@
{
    python $repo/scripts/label-audio.py \
        --tsv_path CSTR-Edinburgh/vctk \
        --dataset-type vctk \
        --out_path $dsus \
        --batch-size 4 \
        --num-workers 0 \
        --ckpt_path $hubert_model \
        --layer $hubert_layer \
        --km_path $km_model \
        --no-dedup
}

# this one stays as is
task LabelCommonvoice
    > dsus=dsus_no_dedup.txt
    :: hubert_model=@
    :: hubert_layer=@
    :: km_model=@
    :: hubert_dtype=@
    :: commonvoice_path=@
    :: repo=@
{
    python $repo/scripts/label-audio.py \
        --tsv_path $commonvoice_path \
        --dataset-type commonvoice \
        --out_path $dsus \
        --batch-size 4 \
        --num-workers 0 \
        --ckpt_path $hubert_model \
        --layer $hubert_layer \
        --km_path $km_model \
        --dtype $hubert_dtype \
        --no-dedup
}

task DeduplicateVCTKLabels
    < dsus_no_dedup=$dsus@LabelVCTK
    > dsus=dsus.txt
    :: repo=@
{
    python $repo/scripts/deduplicate.py < $dsus_no_dedup > $dsus
}

task BuildVCTKInstructions
    < test_dsus=$dsus@DeduplicateVCTKLabels
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

task DecodeVCTK
    < instructions=@BuildVCTKInstructions
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

task LabelAudio
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
        --dtype $hubert_dtype
}

# obsolete, but not deleting just yet
task LabelAudioFromTSV
    < wav_tsv=(ASRDataset: ls_clean=$wav_tsv@ConvertTSV ls_other=$wav_tsv@ConvertTSV fleurs=$tsv@BuildNativeTSV voxpopuli=$tsv@BuildNativeTSV)
    > dsus=dsu.txt
    :: hubert_model=@
    :: hubert_layer=@
    :: km_model=@
    :: hubert_dtype=@
    :: repo=@
{
    python $repo/scripts/label-audio.py \
        --tsv_path $wav_tsv \
        --out_path $dsus \
        --ckpt_path $hubert_model \
        --layer $hubert_layer \
        --km_path $km_model \
        --batch-size 1000000 \
        --dtype $hubert_dtype
}

task LabelAudioDupe
    < wav_tsv=(ASRDataset: ls_clean=$wav_tsv@ConvertTSV ls_other=$wav_tsv@ConvertTSV fleurs=$tsv@BuildNativeTSV voxpopuli=$tsv@BuildNativeTSV)
    > dsus=dsu.txt
    :: hubert_model=@
    :: hubert_layer=@
    :: km_model=@
    :: hubert_dtype=@
    :: repo=@
{
    python $repo/scripts/label-audio.py \
        --tsv_path $wav_tsv \
        --out_path $dsus \
        --ckpt_path $hubert_model \
        --layer $hubert_layer \
        --km_path $km_model \
        --batch-size 1000000 \
        --dtype $hubert_dtype \
        --no-dedup
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

task DecodeSeamless
    < wav_tsv=(ASRDataset: ls_clean=$wav_tsv@ConvertTSV ls_other=$wav_tsv@ConvertTSV fleurs=$tsv@BuildNativeTSV voxpopuli=$tsv@BuildNativeTSV)
    > hyps=hyps.txt
    :: speech2text_model=@
    :: speech2text_tokenizer=@
    :: max_tokens=@
    :: seamless_batch_size=@
    :: repo=@
{
    python $repo/scripts/inference.py \
        --inpaths $wav_tsv \
        --outpaths $hyps \
        --model $speech2text_model \
        --tokenizer $speech2text_tokenizer \
        --backend hf \
        --max-length $max_tokens \
        --src-lang eng \
        --tgt-lang eng \
        --input-format wav_tsv \
        --batch-size $seamless_batch_size \
        --model-type encdec-speech \
        --backend hf \
        --seamless-input-type speech

}

task DecodeWhisper
    < wav_tsv=(ASRDataset: ls_clean=$wav_tsv@ConvertTSV ls_other=$wav_tsv@ConvertTSV fleurs=$tsv@BuildNativeTSV voxpopuli=$tsv@BuildNativeTSV)
    > hyps=hyps.txt
    :: speech2text_model=@
    :: speech2text_tokenizer=@
    :: max_tokens=@
    :: whisper_batch_size=@
    :: repo=@
{
    python $repo/scripts/whisper_inference.py \
        --input-tsv $wav_tsv \
        --outpath $hyps \
        --model $speech2text_model \
        --tokenizer $speech2text_tokenizer \
        --max-length $max_tokens \
        --batch-size $whisper_batch_size
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
    < hyps=(ModelFamily: tower=$hyps@Decode seamless=$hyps@DecodeSeamless whisper=$hyps@DecodeWhisper)
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

##### VCTK Experiments #####

plan LabelizeVCTK {
    reach LabelVCTK
}

plan VCTKInstructions {
    reach BuildVCTKInstructions via (Speech2TextModel: spire_full)
}

plan DecodeVCTKSpireFull {
    reach DecodeVCTK via (ASRDataset: *) * (ModelFamily: tower) * (Speech2TextModel: spire_full)
}

plan DecodeVCTK {
    reach DecodeVCTK via (ASRDataset: *) * (ModelFamily: tower) * (Speech2TextModel: *)
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

plan LabelizeLegacy {
    reach LabelAudioLegacy via (ASRDataset: *)
}

plan LabelizeNew {
    reach LabelAudioNew via (ASRDataset: *)
}

##### Hypothesis Generation #####

plan DecodeAll {
    reach Decode via (ASRDataset: *) * (ModelFamily: tower) * (Speech2TextModel: *)
}

plan DecodeSpireFull {
    reach Decode via (ASRDataset: *) * (ModelFamily: tower) * (Speech2TextModel: spire_full)
}

plan DecodeSpireFullBF16 {
    reach Decode via (ASRDataset: *) * (ModelFamily: tower) * (Speech2TextModel: spire_full) * (HubertDtype: bf16)
}

plan DecodeTowerSpire {
    reach Decode via (ASRDataset: *) * (ModelFamily: tower) * (Speech2TextModel: spire_base spire_full spire_no_pseudo spire_no_blocks tower_full)
}

plan DecodeSL {
    reach DecodeSeamless via (ASRDataset: *) * (ModelFamily: seamless) * (Speech2TextModel: seamless)
}

plan DecodeWh {
    reach DecodeWhisper via (ASRDataset: *) * (ModelFamily: whisper) * (Speech2TextModel: whisper whisper_medium whisper_small whisper_base)
}

##### Evaluation #####

plan ComputeWERAll {
    reach ComputeWER via (ASRDataset: *) * (ModelFamily: tower) * (Speech2TextModel: *)
}

plan ComputeWERSpireFull {
    reach ComputeWER via (ASRDataset: *) * (ModelFamily: tower) * (Speech2TextModel: spire_full)
}

plan ComputeWERSpireFullBF16 {
    reach ComputeWER via (ASRDataset: *) * (ModelFamily: tower) * (Speech2TextModel: spire_full) * (HubertDtype: bf16)
}

plan ComputeWERTowerSpire {
    reach ComputeWER via (ASRDataset: *) * (ModelFamily: tower) * (Speech2TextModel: spire_base spire_full spire_no_pseudo spire_no_blocks tower_full)
}

plan ComputeSeamlessWER {
    reach ComputeWER via (ASRDataset: *) * (ModelFamily: seamless) * (Speech2TextModel: seamless)
}

plan ComputeWhisperWER {
    reach ComputeWER via (ASRDataset: *) * (ModelFamily: whisper) * (Speech2TextModel: whisper whisper_medium whisper_small whisper_base)
}

##### Commonvoice DSU labeling (for analysis)

plan CVFP32 {
    reach LabelCommonvoice via (HubertDtype: fp32)
}

plan CVBF16 {
    reach LabelCommonvoice via (HubertDtype: bf16)
}

