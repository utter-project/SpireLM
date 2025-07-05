task LabelCorpus
    > dsus=dsus_no_dedup.txt
    :: hubert_model=@
    :: hubert_layer=@
    :: km_model=@
    :: hubert_dtype=@
    :: dataset_path=@
    :: dataset_type=@
    :: path_extra=@
    :: shard_number=@
    :: shard_size=@
    :: hf_split=@
    :: batch_size=(BatchSize: 4 1)
    :: repo=@
{
    start_ix=$(( ($shard_number - 1) * $shard_size))
    python $repo/scripts/label-audio.py \
        --tsv_path $dataset_path \
        --dataset-type $dataset_type \
        --out_path $dsus \
        --batch-size $batch_size \
        --num-workers 0 \
        --ckpt_path $hubert_model \
        --layer $hubert_layer \
        --km_path $km_model \
        --dtype $hubert_dtype \
        --start-ix $start_ix \
        --n-examples $shard_size \
        --validate-examples \
        --hf-split $hf_split \
        --resample-to 16000 \
        --path-extra $path_extra \
        --dataset-type $dataset_type \
        --no-dedup
}

task Deduplicate
    < dsus=@LabelCorpus
    > dsus_dedup=dsus.txt
    :: repo=@
{
    python $repo/scripts/deduplicate.py < $dsus > $dsus_dedup
}

plan LabelFullPrecisionAll {
    reach LabelCorpus via (ShardNumber: *)
}

plan DeduplicateAll {
    reach Deduplicate via (ShardNumber: *)
}

plan DeduplicateBaselineB1 {
    reach Deduplicate via (ShardNumber: 1) * (BatchSize: 1)
}

plan Deduplicate2 {
    reach Deduplicate via (ShardNumber: 2)
}

plan LabelFullPrecisionBaseline {
    reach LabelCorpus via (ShardNumber: 1)
}

plan LabelFullPrecisionBaselineB1 {
    reach LabelCorpus via (ShardNumber: 1) * (BatchSize: 1)
}

plan LabelFullPrecision2 {
    reach LabelCorpus via (ShardNumber: 2)
}

plan LabelFullPrecision11 {
    reach LabelCorpus via (ShardNumber: 11)
}

plan LabelFullPrecision21 {
    reach LabelCorpus via (ShardNumber: 21)
}

plan LabelFullPrecision32 {
    reach LabelCorpus via (ShardNumber: 32)
}

plan LabelFullPrecision33 {
    reach LabelCorpus via (ShardNumber: 33)
}

plan LabelFullPrecision41 {
    reach LabelCorpus via (ShardNumber: 41)
}

plan LabelFullPrecision61 {
    reach LabelCorpus via (ShardNumber: 61)
}

plan LabelFullPrecision1_3 {
    reach LabelCorpus via (ShardNumber: 1..3)
}

plan LabelFullPrecision4_6 {
    reach LabelCorpus via (ShardNumber: 4..6)
}

plan LabelFullPrecision7_9 {
    reach LabelCorpus via (ShardNumber: 7..9)
}

plan LabelFullPrecision10_11 {
    reach LabelCorpus via (ShardNumber: 10..11)
}
