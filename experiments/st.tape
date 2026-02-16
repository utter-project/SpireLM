task DumpSTGoldTranscriptions
    > gold=gold.txt
    > metadata=metadata.json
    :: st_dataset=@
    :: st_path_extra=@
    :: dataset_type=@
    :: st_split=@
    :: st_text_field=@
    :: repo=@
{
    python $repo/scripts/extract_asr_text.py \
        --path $st_dataset --path-extra $st_path_extra --split $st_split --text-field $st_text_field \
        --dataset-type $dataset_type --corpus $gold --metadata $metadata
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
    :: st_dataset=@
    :: path_extra=@
    :: dataset_type=@
    :: repo=@
{
    # dataset_path should be defined in the uservars file
    # dataset_dype should be hf-disk
    # resample-to should not be necessary
    # batch-size should be 4 (since it isn't token-batching anymore)
    python $repo/scripts/label-audio.py \
        --tsv_path $st_dataset \
        --dataset-type $dataset_type \
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
    :: speech2text_tokenizer=@
    :: template=@
    :: language_pair=@
    :: repo=@
{
    # template, language pair
    python $repo/scripts/build_instructions.py \
        --src $test_dsus \
        --src-names dsu_seq \
        --templates $template \
        --template-key $language_pair \
        --tokenizer $speech2text_tokenizer \
        --out $instructions
}

task DecodeDirect
    < instructions=@BuildDirectInstructions
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

task Transcribe
    < instructions=@BuildASRInstructions
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

##### Giga20 Experiments #####

plan DirectTranslationGiga20 {
    reach DecodeDirect via (Speech2TextModel: lmt_4B_giga20h_2000) * (HubertDtype: bf16) * (KMModel: cv20h) * (LanguagePair: en_de en_es en_fr en_it en_ko en_nl en_pt en_ru en_zh)
}

plan EvaluationCV20 {
    reach CometDirect, BleuDirect via (Speech2TextModel: lmt_4B_giga20h_2000) * (HubertDtype: bf16) * (KMModel: cv20h) * (LanguagePair: en_de en_es en_fr en_it en_ko en_nl en_pt en_ru en_zh)
}

##### CV20 Experiments #####

plan DirectTranslationCV20 {
    reach DecodeDirect via (Speech2TextModel: lmt_4B_cv20h_1200 lmt_4B_cv20h_2000) * (HubertDtype: bf16) * (KMModel: cv20h) * (LanguagePair: en_de en_es en_fr en_it en_ko en_nl en_pt en_ru en_zh)
}

plan EvaluationCV20 {
    reach CometDirect, BleuDirect via (Speech2TextModel: lmt_4B_cv20h_1200 lmt_4B_cv20h_2000) * (HubertDtype: bf16) * (KMModel: cv20h) * (LanguagePair: en_de en_es en_fr en_it en_ko en_nl en_pt en_ru en_zh)
}

##### Tokenize audio for ST experiments #####

plan Labelize {
    reach LabelAudio
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
