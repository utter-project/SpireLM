task JoinDSUs
    > dsus=dsu.txt
    :: corpus_dir=@
    :: shard_count=@
    :: repo=@
{
    python $repo/scripts/join_dsu_files.py --corpus-dir $corpus_dir --shard-end $shard_count --out $dsus
}

task FilterAndSubsample
    < dsus=@JoinDSUs
    > st_instructions=instructions.jsonl
    > metadata=metadata.jsonl
    > comet_score=comet-mean.json
    :: threshold=@
    :: mt_corpus=@
    :: wmt_src=@
    :: wmt_tgt=@
    :: tgt_name=@
    :: n_lines=@
    :: subsampling=@
    :: models=@
    :: audio_lengths=@
    :: seed=@
    :: repo=@
{
    pair_corpus="${mt_corpus}/${wmt_src}-${wmt_tgt}.jsonl"
    python $repo/scripts/filter_spite.py \
        --mt-corpus $pair_corpus \
        --filtered-mt-corpus $st_instructions \
        --metadata $metadata \
        --speech-corpus $dsus \
        --threshold $threshold \
        --tgt $tgt_name \
        --seed $seed \
        --n-examples $n_lines \
        --subsampling $subsampling \
        --audio-lengths $audio_lengths \
        --models $models

    python $repo/scripts/mean_comet.py --path $metadata --mean-results $comet_score --jsonl
}

task FilterCorpus
    < dsus=@JoinDSUs
    > st_instructions=instructions.jsonl
    :: threshold=@
    :: mt_corpus=@
    :: wmt_src=@
    :: wmt_tgt=@
    :: tgt_name=@
    :: repo=@
{
    pair_corpus="${mt_corpus}/${wmt_src}-${wmt_tgt}.jsonl"
    python $repo/scripts/filter_spite.py \
        --mt-corpus $pair_corpus \
        --filtered-mt-corpus $st_instructions \
        --speech-corpus $dsus \
        --threshold $threshold \
        --tgt $tgt_name
}

task SubsampleCorpus
    < st_instructions=@FilterCorpus
    > st_instructions_out=instructions.jsonl
    :: n_lines=@
    :: seed=@
    :: repo=@
{
    python $repo/scripts/subsample.py --in-file $st_instructions --out-file $st_instructions_out --n-lines $n_lines --seed $seed
}

#####

plan JoinDSUFiles {
    reach JoinDSUs via (Corpus: *)
}

plan Filter {
    reach FilterCorpus via (Corpus: *) * (LanguagePair: en_de en_es en_fr en_it en_ko en_nl en_pt en_ru en_zh) * (Threshold: *)
}

plan Subsample {
    reach SubsampleCorpus via (Corpus: *) * (LanguagePair: en_de en_es en_fr en_it en_ko en_nl en_pt en_ru en_zh) * (NLines: 250000)
}

plan Subsample_86 {
    reach SubsampleCorpus via (Corpus: *) * (LanguagePair: en_de en_es en_fr en_it en_ko en_nl en_pt en_ru en_zh) * (NLines: 250000) * (Threshold: 0.86)
}

plan Subsample_500k {
    reach SubsampleCorpus via (Corpus: *) * (LanguagePair: en_de en_es en_fr en_it en_ko en_nl en_pt en_ru en_zh) * (NLines: 500000)
}


plan Subsample_1m {
    reach SubsampleCorpus via (Corpus: *) * (LanguagePair: en_de en_es en_fr en_it en_ko en_nl en_pt en_ru en_zh) * (NLines: 1000000)
}

# I think these two should recover the original behavior...

plan DummyThreshold {
    reach FilterAndSubsample via (Corpus: commonvoice) * (LanguagePair: en_pt) * (NLines: 0) * (Threshold: 0.0)
}

plan KeepAllAt85 {
    reach FilterAndSubsample via (Corpus: *) * (LanguagePair: en_de en_es en_fr en_it en_ko en_nl en_pt en_ru en_zh) * (NLines: 0) * (Threshold: 0.85)
}

plan FilterSubsample_250k {
    reach FilterAndSubsample via (Corpus: *) * (LanguagePair: en_de en_es en_fr en_it en_ko en_nl en_pt en_ru en_zh) * (NLines: 250000)
}

plan FilterSubsample_500k {
    reach FilterAndSubsample via (Corpus: *) * (LanguagePair: en_de en_es en_fr en_it en_ko en_nl en_pt en_ru en_zh) * (NLines: 500000)
}

plan FilterSubsample_500k_topk {
    reach FilterAndSubsample via (Corpus: *) * (LanguagePair: en_de en_es en_fr en_it en_ko en_nl en_pt en_ru en_zh) * (NLines: 500000) * (Subsampling: topk) * (Threshold: 0.0)
}

plan FilterSubsample_1m {
    reach FilterAndSubsample via (Corpus: *) * (LanguagePair: en_de en_es en_fr en_it en_ko en_nl en_pt en_ru en_zh) * (NLines: 1000000)
}
