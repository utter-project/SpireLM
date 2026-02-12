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
    :: ssl_batch_size=@
    :: repo=@
{
    start_ix=$(( ($shard_number - 1) * $shard_size))
    python $repo/scripts/label-audio.py \
        --tsv_path $dataset_path \
        --dataset-type $dataset_type \
        --out_path $dsus \
        --batch-size $ssl_batch_size \
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

task LabelCorpusPooled
    < times=@ComputeLengths
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
    :: batch_seconds=@
    :: pooling_width=@
    :: pooling_type=@
    :: repo=@
{
    start_ix=$(( ($shard_number - 1) * $shard_size))
    extra_args=""
    if [[ $path_extra != "" ]] ; then
        extra_args="--path-extra ${path_extra}"
    fi
    python $repo/scripts/label-audio.py \
        --tsv_path $dataset_path \
        --dataset-type $dataset_type \
        --out_path $dsus \
        --batch-size $batch_seconds \
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
        --dataset-type $dataset_type \
        --no-dedup \
        --pooling-width $pooling_width \
        --pooling-type $pooling_type \
        --token-batching \
        --example-lengths $times \
        $extra_args
}

task LabelCorpusImproved
    < times=@ComputeLengths
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
    :: batch_seconds=@
    :: pooling_width=@
    :: pooling_type=@
    :: repo=@
{
    start_ix=$(( ($shard_number - 1) * $shard_size))
    extra_args=""
    if [[ $path_extra != "" ]] ; then
        extra_args="--path-extra ${path_extra}"
    fi
    python $repo/scripts/label-audio.py \
        --tsv_path $dataset_path \
        --dataset-type $dataset_type \
        --out_path $dsus \
        --batch-size $batch_seconds \
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
        --dataset-type $dataset_type \
        --no-dedup \
        --token-batching \
        --example-lengths $times \
        $extra_args
}

task LabelCorpusPooledVanilla
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
    :: ssl_batch_size=@
    :: pooling_width=@
    :: pooling_type=@
    :: repo=@
{
    start_ix=$(( ($shard_number - 1) * $shard_size))
    python $repo/scripts/label-audio.py \
        --tsv_path $dataset_path \
        --dataset-type $dataset_type \
        --out_path $dsus \
        --batch-size $ssl_batch_size \
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
        --no-dedup \
        --pooling-width $pooling_width \
        --pooling-type $pooling_type
}

task ComputeSonarSpeechVectors
    < times=@ComputeLengths
    > sonar_vectors=sonar.npy
    :: dataset_path=@
    :: dataset_type=@
    :: path_extra=@
    :: shard_number=@
    :: shard_size=@
    :: hf_split=@
    :: batch_seconds=@
    :: repo=@
{
    start_ix=$(( ($shard_number - 1) * $shard_size))
    extra_args=""
    if [[ $path_extra != "" ]] ; then
        extra_args="--path-extra ${path_extra}"
    fi
    python $repo/scripts/save-sonar-features.py \
        --sonar-speech-encoder sonar_speech_encoder_eng \
        --data-path $dataset_path \
        --dataset-type $dataset_type \
        --save-vectors $sonar_vectors \
        --batch-size $batch_seconds \
        --num-workers 0 \
        --start-ix $start_ix \
        --n-examples $shard_size \
        --hf-split $hf_split \
        --resample-to 16000 \
        --token-batching \
        --example-lengths $times \
        $extra_args
}

task ComputeLengths
    :: dataset_path=@
    :: path_extra=@
    :: hf_split=@
    :: repo=@
    > times=times.npy
{
    python $repo/scripts/compute_corpus_lengths.py \
        --path $dataset_path --path-extra $path_extra --split $hf_split --out $times
}

task Deduplicate
    < dsus=@LabelCorpus
    > dsus_dedup=dsus.txt
    :: repo=@
{
    python $repo/scripts/deduplicate.py < $dsus > $dsus_dedup
}

task DeduplicatePooled
    < dsus=@LabelCorpusPooled
    > dsus_dedup=dsus.txt
    :: repo=@
{
    python $repo/scripts/deduplicate.py < $dsus > $dsus_dedup
}

task DeduplicateImproved
    < dsus=@LabelCorpusImproved
    > dsus_dedup=dsus.txt
    :: repo=@
{
    python $repo/scripts/deduplicate.py < $dsus > $dsus_dedup
}

plan Lengths {
    reach ComputeLengths
}

##### vctk_mic1_30h

plan DD_VCTK_Mic1_30h {
    reach DeduplicateImproved via (ShardNumber: *) * (HubertDtype: bf16) * (BatchSeconds: 240) * (KMModel: vctk_mic1_30h)
}

plan LabelVCTK_Mic1_30h {
    reach LabelCorpusImproved via (ShardNumber: *) * (HubertDtype: bf16) * (BatchSeconds: 240) * (KMModel: vctk_mic1_30h)
}

plan LabelVCTK_Mic1_30h_1_10 {
    reach LabelCorpusImproved via (ShardNumber: 1..10) * (HubertDtype: bf16) * (BatchSeconds: 240) * (KMModel: vctk_mic1_30h)
}

plan LabelVCTK_Mic1_30h_11_20 {
    reach LabelCorpusImproved via (ShardNumber: 11..20) * (HubertDtype: bf16) * (BatchSeconds: 240) * (KMModel: vctk_mic1_30h)
}

plan LabelVCTK_Mic1_30h_21_30 {
    reach LabelCorpusImproved via (ShardNumber: 21..30) * (HubertDtype: bf16) * (BatchSeconds: 240) * (KMModel: vctk_mic1_30h)
}

plan LabelVCTK_Mic1_30h_31_40 {
    reach LabelCorpusImproved via (ShardNumber: 31..40) * (HubertDtype: bf16) * (BatchSeconds: 240) * (KMModel: vctk_mic1_30h)
}

plan LabelVCTK_Mic1_30h_41_50 {
    reach LabelCorpusImproved via (ShardNumber: 41..50) * (HubertDtype: bf16) * (BatchSeconds: 240) * (KMModel: vctk_mic1_30h)
}

plan LabelVCTK_Mic1_30h_51_60 {
    reach LabelCorpusImproved via (ShardNumber: 51..60) * (HubertDtype: bf16) * (BatchSeconds: 240) * (KMModel: vctk_mic1_30h)
}

plan LabelVCTK_Mic1_30h_61_70 {
    reach LabelCorpusImproved via (ShardNumber: 61..70) * (HubertDtype: bf16) * (BatchSeconds: 240) * (KMModel: vctk_mic1_30h)
}

plan LabelVCTK_Mic1_30h_71_83 {
    reach LabelCorpusImproved via (ShardNumber: 71..83) * (HubertDtype: bf16) * (BatchSeconds: 240) * (KMModel: vctk_mic1_30h)
}

##### giga80h_pool4

plan DDP4 {
    reach DeduplicatePooled via (ShardNumber: *) * (HubertDtype: bf16) * (PoolingWidth: 4) * (BatchSeconds: 240) * (KMModel: giga80h_pool4)
}

plan LabelPooledGiga80hPool4 {
    reach LabelCorpusPooled via (ShardNumber: *) * (HubertDtype: bf16) * (PoolingWidth: 4) * (BatchSeconds: 240) * (KMModel: giga80h_pool4)
}

plan LabelPooledGiga80hPool4_1_10 {
    reach LabelCorpusPooled via (ShardNumber: *) * (HubertDtype: bf16) * (PoolingWidth: 4) * (BatchSeconds: 240) * (KMModel: giga80h_pool4) * (ShardNumber: 1..10)
}

plan LabelPooledGiga80hPool4_11_20 {
    reach LabelCorpusPooled via (ShardNumber: *) * (HubertDtype: bf16) * (PoolingWidth: 4) * (BatchSeconds: 240) * (KMModel: giga80h_pool4) * (ShardNumber: 11..20)
}

plan LabelPooledGiga80hPool4_1_20 {
    reach LabelCorpusPooled via (ShardNumber: *) * (HubertDtype: bf16) * (PoolingWidth: 4) * (BatchSeconds: 240) * (KMModel: giga80h_pool4) * (ShardNumber: 1..20)
}

plan LabelPooledGiga80hPool4_21_40 {
    reach LabelCorpusPooled via (ShardNumber: *) * (HubertDtype: bf16) * (PoolingWidth: 4) * (BatchSeconds: 240) * (KMModel: giga80h_pool4) * (ShardNumber: 21..40)
}

plan LabelPooledGiga80hPool4_41_60 {
    reach LabelCorpusPooled via (ShardNumber: *) * (HubertDtype: bf16) * (PoolingWidth: 4) * (BatchSeconds: 240) * (KMModel: giga80h_pool4) * (ShardNumber: 41..60)
}

plan LabelPooledGiga80hPool4_61_83 {
    reach LabelCorpusPooled via (ShardNumber: *) * (HubertDtype: bf16) * (PoolingWidth: 4) * (BatchSeconds: 240) * (KMModel: giga80h_pool4) * (ShardNumber: 61..83)
}

plan LabelPooledGiga80hPool4_1_3 {
    reach LabelCorpusPooled via (ShardNumber: *) * (HubertDtype: bf16) * (PoolingWidth: 4) * (BatchSeconds: 240) * (KMModel: giga80h_pool4) * (ShardNumber: 1..3)
}

plan LabelPooledGiga80hPool4_4_6 {
    reach LabelCorpusPooled via (ShardNumber: *) * (HubertDtype: bf16) * (PoolingWidth: 4) * (BatchSeconds: 240) * (KMModel: giga80h_pool4) * (ShardNumber: 4..6)
}

plan LabelPooledGiga80hPool4_7_9 {
    reach LabelCorpusPooled via (ShardNumber: *) * (HubertDtype: bf16) * (PoolingWidth: 4) * (BatchSeconds: 240) * (KMModel: giga80h_pool4) * (ShardNumber: 7..9)
}

plan LabelPooledGiga80hPool4_10_12 {
    reach LabelCorpusPooled via (ShardNumber: *) * (HubertDtype: bf16) * (PoolingWidth: 4) * (BatchSeconds: 240) * (KMModel: giga80h_pool4) * (ShardNumber: 10..12)
}

plan LabelPooledGiga80hPool4_13_16 {
    reach LabelCorpusPooled via (ShardNumber: *) * (HubertDtype: bf16) * (PoolingWidth: 4) * (BatchSeconds: 240) * (KMModel: giga80h_pool4) * (ShardNumber: 13..16)
}

#####



plan DeduplicateImprovedGiga30h {
    reach DeduplicateImproved via (ShardNumber: *) * (HubertDtype: bf16) * (BatchSeconds: 240) * (KMModel: giga30h)
}

plan DeduplicateImprovedGiga30h_1_10 {
    reach DeduplicateImproved via (ShardNumber: *) * (HubertDtype: bf16) * (BatchSeconds: 240) * (KMModel: giga30h) * (ShardNumber: 1..10)
}

plan DeduplicateImprovedGiga30h_11_20 {
    reach DeduplicateImproved via (ShardNumber: *) * (HubertDtype: bf16) * (BatchSeconds: 240) * (KMModel: giga30h) * (ShardNumber: 11..20)
}

plan DeduplicateImprovedGiga30h_1_20 {
    reach DeduplicateImproved via (ShardNumber: *) * (HubertDtype: bf16) * (BatchSeconds: 240) * (KMModel: giga30h) * (ShardNumber: 1..20)
}

plan DeduplicateImprovedGiga30h_21_40 {
    reach DeduplicateImproved via (ShardNumber: *) * (HubertDtype: bf16) * (BatchSeconds: 240) * (KMModel: giga30h) * (ShardNumber: 21..40)
}

plan DeduplicateImprovedGiga30h_41_60 {
    reach DeduplicateImproved via (ShardNumber: *) * (HubertDtype: bf16) * (BatchSeconds: 240) * (KMModel: giga30h) * (ShardNumber: 41..60)
}

plan DeduplicateImprovedGiga30h_41_45 {
    reach DeduplicateImproved via (ShardNumber: *) * (HubertDtype: bf16) * (BatchSeconds: 240) * (KMModel: giga30h) * (ShardNumber: 41..45)
}

plan DeduplicateImprovedGiga30h_46_50 {
    reach DeduplicateImproved via (ShardNumber: *) * (HubertDtype: bf16) * (BatchSeconds: 240) * (KMModel: giga30h) * (ShardNumber: 46..50)
}

plan DeduplicateImprovedGiga30h_51_55 {
    reach DeduplicateImproved via (ShardNumber: *) * (HubertDtype: bf16) * (BatchSeconds: 240) * (KMModel: giga30h) * (ShardNumber: 51..55)
}

plan DeduplicateImprovedGiga30h_56_60 {
    reach DeduplicateImproved via (ShardNumber: *) * (HubertDtype: bf16) * (BatchSeconds: 240) * (KMModel: giga30h) * (ShardNumber: 56..60)
}

plan DeduplicateImprovedGiga30h_61_83 {
    reach DeduplicateImproved via (ShardNumber: *) * (HubertDtype: bf16) * (BatchSeconds: 240) * (KMModel: giga30h) * (ShardNumber: 61..83)
}



plan DeduplicateImprovedMHCV20h {
    reach DeduplicateImproved via (ShardNumber: *) * (HubertDtype: bf16) * (BatchSeconds: 240) * (KMModel: mhubert_25hz_cv20h)
}

plan DeduplicateImprovedMHCV20h_1_20 {
    reach DeduplicateImproved via (ShardNumber: *) * (HubertDtype: bf16) * (BatchSeconds: 240) * (KMModel: mhubert_25hz_cv20h) * (ShardNumber: 1..20)
}

plan DeduplicateImprovedMHCV20h_21_40 {
    reach DeduplicateImproved via (ShardNumber: *) * (HubertDtype: bf16) * (BatchSeconds: 240) * (KMModel: mhubert_25hz_cv20h) * (ShardNumber: 21..40)
}

plan DeduplicateImprovedMHCV20h_41_60 {
    reach DeduplicateImproved via (ShardNumber: *) * (HubertDtype: bf16) * (BatchSeconds: 240) * (KMModel: mhubert_25hz_cv20h) * (ShardNumber: 41..60)
}

plan DeduplicateImprovedMHCV20h_61_83 {
    reach DeduplicateImproved via (ShardNumber: *) * (HubertDtype: bf16) * (BatchSeconds: 240) * (KMModel: mhubert_25hz_cv20h) * (ShardNumber: 61..83)
}

plan DeduplicateImprovedCV20h {
    reach DeduplicateImproved via (ShardNumber: *) * (HubertDtype: bf16) * (BatchSeconds: 240) * (KMModel: cv20h)
}

plan DeduplicateImprovedGiga20h {
    reach DeduplicateImproved via (ShardNumber: *) * (HubertDtype: bf16) * (BatchSeconds: 240) * (KMModel: giga20h)
}

plan LabelGiga20h {
    reach LabelCorpusImproved via (ShardNumber: *) * (HubertDtype: bf16) * (BatchSeconds: 240) * (KMModel: giga20h)
}

plan LabelCV20h {
    reach LabelCorpusImproved via (ShardNumber: *) * (HubertDtype: bf16) * (BatchSeconds: 240) * (KMModel: cv20h)
}

plan LabelCV20h_1_10 {
    reach LabelCorpusImproved via (ShardNumber: *) * (HubertDtype: bf16) * (BatchSeconds: 240) * (KMModel: cv20h) * (ShardNumber: 1..10)
}

plan LabelCV20h_11_20 {
    reach LabelCorpusImproved via (ShardNumber: *) * (HubertDtype: bf16) * (BatchSeconds: 240) * (KMModel: cv20h) * (ShardNumber: 11..20)
}

plan LabelCV20h_21_40 {
    reach LabelCorpusImproved via (ShardNumber: *) * (HubertDtype: bf16) * (BatchSeconds: 240) * (KMModel: cv20h) * (ShardNumber: 21..40)
}

plan LabelCV20h_41_60 {
    reach LabelCorpusImproved via (ShardNumber: *) * (HubertDtype: bf16) * (BatchSeconds: 240) * (KMModel: cv20h) * (ShardNumber: 41..60)
}

plan LabelCV20h_61_83 {
    reach LabelCorpusImproved via (ShardNumber: *) * (HubertDtype: bf16) * (BatchSeconds: 240) * (KMModel: cv20h) * (ShardNumber: 61..83)
}

plan LabelGiga20h_1_20 {
    reach LabelCorpusImproved via (ShardNumber: *) * (HubertDtype: bf16) * (BatchSeconds: 240) * (KMModel: giga20h) * (ShardNumber: 1..20)
}

plan LabelGiga20h_21_40 {
    reach LabelCorpusImproved via (ShardNumber: *) * (HubertDtype: bf16) * (BatchSeconds: 240) * (KMModel: giga20h) * (ShardNumber: 21..40)
}

plan LabelGiga20h_41_60 {
    reach LabelCorpusImproved via (ShardNumber: *) * (HubertDtype: bf16) * (BatchSeconds: 240) * (KMModel: giga20h) * (ShardNumber: 41..60)
}

plan LabelGiga20h_61_83 {
    reach LabelCorpusImproved via (ShardNumber: *) * (HubertDtype: bf16) * (BatchSeconds: 240) * (KMModel: giga20h) * (ShardNumber: 61..83)
}

plan LabelPooled {
    reach LabelCorpusPooled via (ShardNumber: *) * (HubertDtype: bf16) * (PoolingWidth: 2) * (BatchSeconds: 240)
}

plan LabelPooled_1_10 {
    reach LabelCorpusPooled via (ShardNumber: *) * (HubertDtype: bf16) * (PoolingWidth: 2) * (BatchSeconds: 240) * (ShardNumber: 1..10)
}

plan LabelPooled_11_20 {
    reach LabelCorpusPooled via (ShardNumber: *) * (HubertDtype: bf16) * (PoolingWidth: 2) * (BatchSeconds: 240) * (ShardNumber: 11..20)
}

plan LabelPooled_21_30 {
    reach LabelCorpusPooled via (ShardNumber: *) * (HubertDtype: bf16) * (PoolingWidth: 2) * (BatchSeconds: 240) * (ShardNumber: 21..30)
}

plan LabelPooled_21_30_no_25 {
    reach LabelCorpusPooled via (ShardNumber: *) * (HubertDtype: bf16) * (PoolingWidth: 2) * (BatchSeconds: 240) * (ShardNumber: 21 22 23 24 26 27 28 29 30)
}

plan LabelPooled_25 {
    reach LabelCorpusPooled via (ShardNumber: *) * (HubertDtype: bf16) * (PoolingWidth: 2) * (BatchSeconds: 240) * (ShardNumber: 25)
}

plan LabelPooled_31_40 {
    reach LabelCorpusPooled via (ShardNumber: *) * (HubertDtype: bf16) * (PoolingWidth: 2) * (BatchSeconds: 240) * (ShardNumber: 31..40)
}

plan LabelPooled_41_50 {
    reach LabelCorpusPooled via (ShardNumber: *) * (HubertDtype: bf16) * (PoolingWidth: 2) * (BatchSeconds: 240) * (ShardNumber: 41..50)
}

plan LabelPooled_51_60 {
    reach LabelCorpusPooled via (ShardNumber: *) * (HubertDtype: bf16) * (PoolingWidth: 2) * (BatchSeconds: 240) * (ShardNumber: 51..60)
}

plan LabelPooled_61_70 {
    reach LabelCorpusPooled via (ShardNumber: *) * (HubertDtype: bf16) * (PoolingWidth: 2) * (BatchSeconds: 240) * (ShardNumber: 61..70)
}

plan LabelPooled_71_83 {
    reach LabelCorpusPooled via (ShardNumber: *) * (HubertDtype: bf16) * (PoolingWidth: 2) * (BatchSeconds: 240) * (ShardNumber: 71..83)
}

plan DDP {
    reach DeduplicatePooled via (ShardNumber: *) * (HubertDtype: bf16) * (PoolingWidth: 2) * (BatchSeconds: 240)
}

plan LabelPooled480 {
    reach LabelCorpusPooled via (ShardNumber: 1) * (HubertDtype: bf16) * (PoolingWidth: 2) * (BatchSeconds: 480)
}

plan LabelPooledVanilla {
    reach LabelCorpusPooledVanilla via (ShardNumber: *) * (HubertDtype: bf16) * (PoolingWidth: 2) * (BatchSize: 4)
}

plan LabelPooledVanilla1 {
    reach LabelCorpusPooledVanilla via (ShardNumber: 1) * (HubertDtype: bf16) * (PoolingWidth: 2) * (BatchSize: 1)
}

plan LabelPooledVanilla16 {
    reach LabelCorpusPooledVanilla via (ShardNumber: *) * (HubertDtype: bf16) * (PoolingWidth: 2) * (BatchSize: 16)
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

##### Compute sonar embeddings for the audio #####

plan Sonar {
    reach ComputeSonarSpeechVectors via (ShardNumber: *) * (BatchSeconds: 480)
}

plan Sonar_1_10 {
    reach ComputeSonarSpeechVectors via (ShardNumber: 1..10) * (BatchSeconds: 480)
}

plan Sonar_11_20 {
    reach ComputeSonarSpeechVectors via (ShardNumber: 11..20) * (BatchSeconds: 480)
}

plan Sonar_21_30 {
    reach ComputeSonarSpeechVectors via (ShardNumber: 21..30) * (BatchSeconds: 480)
}

plan Sonar_21_30x25 {
    reach ComputeSonarSpeechVectors via (ShardNumber: 21 22 23 24 26 27 28 29 30) * (BatchSeconds: 480)
}

plan Sonar_25 {
    reach ComputeSonarSpeechVectors via (ShardNumber: 25) * (BatchSeconds: 480)
}

plan Sonar_31_40 {
    reach ComputeSonarSpeechVectors via (ShardNumber: 31..40) * (BatchSeconds: 480)
}

plan Sonar_41_50 {
    reach ComputeSonarSpeechVectors via (ShardNumber: 41..50) * (BatchSeconds: 480)
}

plan Sonar_51_60 {
    reach ComputeSonarSpeechVectors via (ShardNumber: 51..60) * (BatchSeconds: 480)
}

plan Sonar_61_70 {
    reach ComputeSonarSpeechVectors via (ShardNumber: 61..70) * (BatchSeconds: 480)
}

plan Sonar_71_83 {
    reach ComputeSonarSpeechVectors via (ShardNumber: 71..83) * (BatchSeconds: 480)
}

plan Sonar_10 {
    reach ComputeSonarSpeechVectors via (ShardNumber: 10) * (BatchSeconds: 480)
}

plan Sonar_20 {
    reach ComputeSonarSpeechVectors via (ShardNumber: 20) * (BatchSeconds: 480)
}
