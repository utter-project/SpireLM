task BuildSpiteInstructions
    > instructions=instructions.jsonl
    > stats=stats.json
    :: dsus_for_pseudolabeling=@
    :: dataset=@
    :: language_pair=@
    :: template=@
    :: min_cometqe=@
    :: min_blaser2_src=@
    :: min_audio_length=@
    :: topk_examples=@
    :: topk_column=@
    :: topk_mode=@
    :: repo=@
{
    python $repo/scripts/build_dsu_training.py \
        --dsus $dsus_for_pseudolabeling \
        --dataset-path $dataset \
        --path-extra $language_pair \
        --split train \
        --transcript-column mt \
        --min-columns cometqe_22 blaser2_src audio_length \
        --min-column-values $min_cometqe $min_blaser2_src $min_audio_length \
        --templates $template \
        --template-key $language_pair \
        --instructions $instructions \
        --topk-examples $topk_examples \
        --topk-column $topk_column \
        --topk-mode $topk_mode \
        --spite-stats $stats
}

task BuildSpiteASRInstructions
    > instructions=instructions.jsonl
    :: dsus_for_pseudolabeling=@
    :: dataset=@
    :: template=@
    :: min_cometqe=@
    :: min_blaser2_src=@
    :: min_audio_length=@
    :: repo=@
{
    # any arbitrary language pair choice works for path-extra
    python $repo/scripts/build_dsu_training.py \
        --dsus $dsus_for_pseudolabeling \
        --dataset-path $dataset \
        --path-extra en_de \
        --split train \
        --transcript-column src \
        --min-columns blaser2_src audio_length \
        --min-column-values $min_blaser2_src $min_audio_length \
        --templates $template \
        --template-key asr \
        --instructions $instructions
}

plan BuildSpite {
    reach BuildSpiteInstructions via (CorpusForPseudolabeling: *) * (Model: *) * (LanguagePair: *)
}

plan BuildSpiteTP {
    reach BuildSpiteInstructions via (CorpusForPseudolabeling: *) * (Model: tp) * (LanguagePair: *)
}

plan BuildSpiteTPTopComet {
    reach BuildSpiteInstructions via (CorpusForPseudolabeling: *) * (Model: tp) * (LanguagePair: *)
}

plan BuildCVASR {
    reach BuildSpiteASRInstructions via (CorpusForPseudolabeling: commonvoice)
}

plan BuildSpiteExplicit {
    reach BuildSpiteInstructions via (CorpusForPseudolabeling: *) * (Model: *) * (LanguagePair: *) * (Template: explicit)
}

plan BuildSpiteTPExplicit {
    reach BuildSpiteInstructions via (CorpusForPseudolabeling: *) * (Model: tp) * (LanguagePair: *) * (Template: explicit)
}

plan BuildCVASRExplicit {
    reach BuildSpiteASRInstructions via (CorpusForPseudolabeling: commonvoice) * (Template: explicit)
}

##### Here: preliminary filtering experiment. Constant number of examples, gigaspeech en-de only

plan BuildSpiteTPTopK {
    reach BuildSpiteInstructions via (CorpusForPseudolabeling: gigaspeech) * (Model: tp) * (LanguagePair: en_de) * (TopK: *) * (TopKColumn: *)
}
