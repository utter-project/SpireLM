task DumpSrcRef
    > src=src.txt
    > ref=ref.txt
    :: dataset_config=@
    :: transcript_field=@
    :: translation_field=@
    :: external_ref_path=@
    :: flores_src=@
    :: flores_tgt=@
    :: repo=@
{
    # goal of this task: unify FetchFlores, DumpSTGoldTranscriptions,
    # DumpSTReferenceTranslations, and RecoverReferencesFleurs

    # The external_ref_path is the path from which flores can be downloaded.
    # It is necessary for fleurs but not CoVoST-2. If doing a CoVoST-2 config,
    # external_ref_path should be defined but can be the empty string.

    # On the other hand, translation_field is only used for CoVoST-2 and can
    # be left blank for fleurs. I could combined these into a single variable,
    # but it seemed a bit inelegant to do so.

    full_dataset_config=$dataset_config
    if [ ! -f $full_dataset_config ] ; then
        full_dataset_config="${repo}/${full_dataset_config}"
    fi

    python $repo/scripts/extract_asr_text.py \
        --config $full_dataset_config \
        --text-field $transcript_field \
        --corpus $src --metadata src_metadata.tmp


    if [[ $translation_field != "" ]] ; then
        python $repo/scripts/extract_asr_text.py \
            --config $full_dataset_config \
            --text-field $translation_field \
            --corpus $ref --metadata ref_metadata.tmp
    else
        # Fetch flores
        python $repo/scripts/hf2text.py --dev dev --devtest devtest --langs $flores_src $flores_tgt
        # use fleurs srcs to find aligned translations in flores
        python $repo/scripts/build_fleurs_references.py \
            --flores-src "devtest/devtest.${flores_src}" \
            --flores-tgt "devtest/devtest.${flores_tgt}" \
            --fleurs-src $src --fleurs-tgt-inferred $ref
    fi
}

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
    if [ ! -f $full_dataset_config ] ; then
        full_dataset_config="${repo}/${full_dataset_config}"
    fi

    python $repo/scripts/label-audio.py \
        --config $full_dataset_config \
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
    if [ ! -f $full_template ] ; then
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
    < src=@DumpSrcRef
    < references=$ref@DumpSrcRef
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
    < references=$ref@DumpSrcRef
    > bleu=bleu.json
    :: repo=@
{
    sacrebleu --metrics bleu chrf --tokenize=flores200 $references < $hyps > $bleu
}

##### Get src and ref for all language pairs #####

plan GetData {
    reach DumpSrcRef via (LanguagePair: *)
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
