task DumpSTGoldTranscriptions
    > gold=gold.txt
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
        --config $repo/$full_dataset_config \
        --text-field $transcript_field \
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
    :: dataset_config=@
    :: repo=@
{
    full_dataset_config=$dataset_config
    if [ ! -f $template ] ; then
        full_dataset_config="${repo}/${full_dataset_config}"
    fi

    python $repo/scripts/label-audio.py \
        --config $repo/$full_dataset_config \
        --out-path $dsus \
        --ssl-model $hubert_model \
        --layer $hubert_layer \
        --kmeans-model $km_model \
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
    full_template=$template
    if [ ! -f $template ] ; then
        full_template="${repo}/${full_template}"
    fi

    # template, language pair
    python $repo/scripts/build_instructions.py \
        --src $test_dsus \
        --src-names dsu_seq \
        --templates $full_template \
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

##### Tokenize audio for ST experiments #####

plan Labelize {
    reach LabelAudio via (HubertDtype: bf16)
}

##### Direct ST #####

plan DirectTranslationAll {
    reach DecodeDirect via (Speech2TextModel: *) * (LanguagePair: *) * (HubertDtype: bf16) * (Template: *)
}

plan EvaluationAll {
    reach CometDirect, BleuDirect via (Speech2TextModel: *) * (LanguagePair: *) * (HubertDtype: bf16) * (Template: *)
}
