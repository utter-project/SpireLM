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

# I would prefer if the sonar/blaser tasks were not in this file, but it was
# convenient
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

task ComputeBlaser2
    < sonar_vectors=@ComputeSonarSpeechVectors
    > scores=blaser2.npy
    :: spite_dataset=@
    :: language_pair=@
    :: sonar_tgt=@
    :: shard_number=@
    :: shard_size=@
    :: blaser2_batch_size=@
    :: repo=@
{
    start_ix=$(( ($shard_number - 1) * $shard_size))
    python $repo/scripts/compute_blaser2.py \
        --dataset-path $spite_dataset \
        --dataset-path-extra $language_pair \
        --lang $sonar_tgt \
        --start-ix $start_ix \
        --n-examples $shard_size \
        --audio-sonar $sonar_vectors \
        --batch-size $blaser2_batch_size \
        --out_path $scores
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

plan Blaser2 {
    reach ComputeBlaser2 via (ShardNumber: *) * (SpiteDataset: *) * (BatchSeconds: 480) * (LanguagePair: en_de en_es en_fr en_it en_ko en_nl en_pt en_ru en_zh)
}

plan Blaser2_1_10 {
    reach ComputeBlaser2 via (ShardNumber: 1..10) * (BatchSeconds: 480) * (LanguagePair: en_de en_es en_fr en_it en_ko en_nl en_pt en_ru en_zh)
}

plan Blaser2_11_20 {
    reach ComputeBlaser2 via (ShardNumber: 11..20) * (BatchSeconds: 480) * (LanguagePair: en_de en_es en_fr en_it en_ko en_nl en_pt en_ru en_zh)
}

plan Blaser2_21_30 {
    reach ComputeBlaser2 via (ShardNumber: 21..30) * (BatchSeconds: 480) * (LanguagePair: en_de en_es en_fr en_it en_ko en_nl en_pt en_ru en_zh)
}

plan Blaser2_31_40 {
    reach ComputeBlaser2 via (ShardNumber: 31..40) * (BatchSeconds: 480) * (LanguagePair: en_de en_es en_fr en_it en_ko en_nl en_pt en_ru en_zh)
}

plan Blaser2_41_50 {
    reach ComputeBlaser2 via (ShardNumber: 41..50) * (BatchSeconds: 480) * (LanguagePair: en_de en_es en_fr en_it en_ko en_nl en_pt en_ru en_zh)
}

plan Blaser2_51_60 {
    reach ComputeBlaser2 via (ShardNumber: 51..60) * (BatchSeconds: 480) * (LanguagePair: en_de en_es en_fr en_it en_ko en_nl en_pt en_ru en_zh)
}

plan Blaser2_61_70 {
    reach ComputeBlaser2 via (ShardNumber: 61..70) * (BatchSeconds: 480) * (LanguagePair: en_de en_es en_fr en_it en_ko en_nl en_pt en_ru en_zh)
}

plan Blaser2_71_83 {
    reach ComputeBlaser2 via (ShardNumber: 71..83) * (BatchSeconds: 480) * (LanguagePair: en_de en_es en_fr en_it en_ko en_nl en_pt en_ru en_zh)
}

##### Now with EuroLLM

plan Blaser2_1_10_euro {
    reach ComputeBlaser2 via (ShardNumber: 1..10) * (BatchSeconds: 480) * (LanguagePair: en_de en_es en_fr en_it en_ko en_nl en_pt en_ru en_zh) * (SpiteDataset: euro)
}

plan Blaser2_11_20_euro {
    reach ComputeBlaser2 via (ShardNumber: 11..20) * (BatchSeconds: 480) * (LanguagePair: en_de en_es en_fr en_it en_ko en_nl en_pt en_ru en_zh) * (SpiteDataset: euro)
}

plan Blaser2_21_30_euro {
    reach ComputeBlaser2 via (ShardNumber: 21..30) * (BatchSeconds: 480) * (LanguagePair: en_de en_es en_fr en_it en_ko en_nl en_pt en_ru en_zh) * (SpiteDataset: euro)
}

plan Blaser2_31_40_euro {
    reach ComputeBlaser2 via (ShardNumber: 31..40) * (BatchSeconds: 480) * (LanguagePair: en_de en_es en_fr en_it en_ko en_nl en_pt en_ru en_zh) * (SpiteDataset: euro)
}

plan Blaser2_41_50_euro {
    reach ComputeBlaser2 via (ShardNumber: 41..50) * (BatchSeconds: 480) * (LanguagePair: en_de en_es en_fr en_it en_ko en_nl en_pt en_ru en_zh) * (SpiteDataset: euro)
}

plan Blaser2_51_60_euro {
    reach ComputeBlaser2 via (ShardNumber: 51..60) * (BatchSeconds: 480) * (LanguagePair: en_de en_es en_fr en_it en_ko en_nl en_pt en_ru en_zh) * (SpiteDataset: euro)
}

plan Blaser2_61_70_euro {
    reach ComputeBlaser2 via (ShardNumber: 61..70) * (BatchSeconds: 480) * (LanguagePair: en_de en_es en_fr en_it en_ko en_nl en_pt en_ru en_zh) * (SpiteDataset: euro)
}

plan Blaser2_71_83_euro {
    reach ComputeBlaser2 via (ShardNumber: 71..83) * (BatchSeconds: 480) * (LanguagePair: en_de en_es en_fr en_it en_ko en_nl en_pt en_ru en_zh) * (SpiteDataset: euro)
}
