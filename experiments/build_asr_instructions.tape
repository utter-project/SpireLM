task BuildASRInstructions
    > instructions=instructions.jsonl
    :: dsus_for_asr=@
    :: dataset=@
    :: path_extra=@
    :: template=@
    :: transcript_column=@
    :: external_transcript_column=@
    :: external_transcript=@
    :: repo=@
{
    extra_args=""
    if [[ $external_transcript != "" ]] ; then
        extra_args="--external-transcripts ${external_transcript}"
    fi
    python $repo/scripts/build_dsu_training.py \
        --dsus $dsus_for_asr \
        --dataset-path $dataset \
        --path-extra $path_extra \
        --audio-dataset \
        --split train \
        --transcript-column $transcript_column \
        --external-transcript-column $external_transcript_column \
        --templates $template \
        --template-key asr \
        --instructions $instructions \
        $extra_args
}

plan BuildASR {
    reach BuildASRInstructions via (Corpus: *)
}
