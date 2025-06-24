task BuildNativeTSV
    :: st_dataset=@
    :: st_path_extra=@
    :: st_split=@
    :: repo=@
    > tsv=audio.tsv
{
    # build a tsv file for an ASR dataset
    python $repo/scripts/build_tsv.py --path $st_dataset --path-extra $st_path_extra --split $st_split --out $tsv
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

# will need to rewrite this one, I think
task DumpSTGoldTranscriptions
    > gold=gold.txt
    > metadata=metadata.json
    :: st_dataset=@
    :: st_path_extra=@
    :: st_split=@
    :: st_text_field=@
    :: repo=@
{
    python $repo/scripts/extract_asr_text.py \
        --path $st_dataset --path-extra $st_path_extra --split $st_split --text-field $st_text_field \
        --corpus $gold --metadata $metadata
}

task FetchFlores
    > dev=dev
    > devtest=devtest
    :: flores_path=@
    :: repo=@
{
    python $repo/scripts/hf2text.py --dev $dev --devtest $devtest
}

task RecoverReferencesFleurs
    < devtest=@FetchFlores
    < gold_src=$gold@DumpSTGoldTranscriptions
    > references=ref.txt
    :: flores_src=@
    :: flores_tgt=@
    :: repo=@
{
    # assume the src is always English
    python $repo/scripts/build_fleurs_references.py \
        --flores-src "${devtest}/devtest.${flores_src}" \
        --flores-tgt "${devtest}/devtest.${flores_tgt}" \
        --fleurs-src $gold_src --fleurs-tgt-inferred $references
}

# the tsv is not needed, so what do we do?
task LabelAudio
    > dsus=dsu.txt
    :: hubert_model=@
    :: hubert_layer=@
    :: km_model=@
    :: hubert_dtype=@
    :: dataset_path=@
    :: path_extra=@
    :: repo=@
{
    # dataset_path should be defined in the uservars file
    # dataset_dype should be hf-disk
    # resample-to should not be necessary
    # batch-size should be 4 (since it isn't token-batching anymore)
    python $repo/scripts/label-audio.py \
        --tsv_path $dataset_path \
        --dataset-type "hf-disk" \
        --path-extra $path_extra \
        --hf-split test \
        --out_path $dsus \
        --ckpt_path $hubert_model \
        --layer $hubert_layer \
        --km_path $km_model \
        --batch-size 4 \
        --dtype $hubert_dtype
}

##### Direct ST tasks #####

task BuildDirectInstructions
    < test_dsus=$dsus@LabelAudio
    > instructions=instructions.json
    :: st_template=@
    :: speech2text_tokenizer=@
    :: tgt_name=@
    :: repo=@
{
    # template, language pair, number of shots
    python $repo/scripts/build_instructions.py \
        --src $test_dsus \
        --template $st_template \
        --n-shots 0 \
        --chat-tokenizer $speech2text_tokenizer \
        --tgt-lang $tgt_name \
        --out $instructions
}

task DecodeDirect
    < instructions=@BuildDirectInstructions
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

task CometDirect
    < hyps=@DecodeDirect
    < src=$gold@DumpSTGoldTranscriptions
    < references=@RecoverReferencesFleurs
    > comet=comet.json
    > means=comet-means.json
    :: flores_src=@
    :: flores_tgt=@
    :: comet_model=@
    :: repo=@
{
    # compute comet (we can get the other metrics in other tasks?)

    comet-score -s $src -t $hyps -r $references --model $comet_model --to_json $comet

    python $repo/scripts/mean_comet.py --path $comet --mean-results $means
}

task BleuDirect
    < hyps=@DecodeDirect
    < references=@RecoverReferencesFleurs
    > bleu=bleu.json
    :: repo=@
{
    sacrebleu --metrics bleu chrf --tokenize=flores200 $references < $hyps > $bleu
}

##### Self-cascaded ST tasks #####

task BuildASRInstructions
    < test_dsus=$dsus@LabelAudio
    > instructions=instructions.json
    :: speech2text_tokenizer=@
    :: repo=@
{
    # template, language pair, number of shots
    python $repo/scripts/build_instructions.py \
        --src $test_dsus \
        --template asr_simple \
        --n-shots 0 \
        --chat-tokenizer $speech2text_tokenizer \
        --out $instructions
}

task Transcribe
    < instructions=@BuildASRInstructions
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

# since it's a self-cascade, we'll just use the speech2text tokenizer
task BuildSelfCascadedInstructions
    < transcriptions=$hyps@Transcribe
    > instructions=instructions.json
    :: src_name=@
    :: tgt_name=@
    :: speech2text_tokenizer=@
    :: repo=@
{
    # template, language pair, number of shots
    python $repo/scripts/build_instructions.py \
        --src $transcriptions \
        --template mt_zero \
        --n-shots 0 \
        --src-lang $src_name \
        --tgt-lang $tgt_name \
        --chat-tokenizer $speech2text_tokenizer \
        --out $instructions
}

task DecodeCascaded
    < instructions=@BuildSelfCascadedInstructions
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

task CometCascaded
    < hyps=@DecodeCascaded
    < src=$gold@DumpSTGoldTranscriptions
    < references=@RecoverReferencesFleurs
    > comet=comet.json
    > means=comet-means.json
    :: flores_src=@
    :: flores_tgt=@
    :: comet_model=@
    :: repo=@
{
    # compute comet (we can get the other metrics in other tasks?)

    comet-score -s $src -t $hyps -r $references --model $comet_model --to_json $comet

    python $repo/scripts/mean_comet.py --path $comet --mean-results $means
}

task BleuCascaded
    < hyps=@DecodeCascaded
    < references=@RecoverReferencesFleurs
    > bleu=bleu.json
    :: repo=@
{
    sacrebleu --metrics bleu chrf --tokenize=flores200 $references < $hyps > $bleu
}

##### Tokenize audio for ST experiments #####

plan Labelize {
    reach LabelAudio via (STDataset: *)
}

##### Prep data #####

plan DirectInstructions {
    reach BuildDirectInstructions via (Speech2TextModel: *) * (LanguagePair: en_de en_es en_fr en_it en_ko en_nl en_pt en_ru en_zh)
}

##### Direct ST #####

plan DirectTranslation {
    reach DecodeDirect via (Speech2TextModel: *) * (LanguagePair: en_de en_es en_fr en_it en_ko en_nl en_pt en_ru en_zh)
}

plan Evaluation {
    reach CometDirect, BleuDirect via (Speech2TextModel: *) * (LanguagePair: en_de en_es en_fr en_it en_ko en_nl en_pt en_ru en_zh)
}

##### Self-cascaded ST #####

plan STTranscription {
    reach Transcribe via (Speech2TextModel: *)
}

plan CascadedTranslation {
    reach DecodeCascaded via (Speech2TextModel: *) * (LanguagePair: en_de en_es en_fr en_it en_ko en_nl en_pt en_ru en_zh)
}

plan CascadedEvaluation {
    reach CometCascaded, BleuCascaded via (Speech2TextModel: *) * (LanguagePair: en_de en_es en_fr en_it en_ko en_nl en_pt en_ru en_zh)
}
