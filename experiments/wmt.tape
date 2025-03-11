task DownloadWMTViaTowerEval
    > wmt
    :: tower_eval_url=@
{
    wget $tower_eval_url
    tar -xzvf data.tar.gz

    mkdir -p $wmt
    cp -r data/raw_data/mt/wmt23.* $wmt
}

task UnpackWMTData
    < wmt=@DownloadWMTViaTowerEval
    > dev_src=dev.src
    > dev_tgt=dev.tgt
    > test_src=test.src
    > test_tgt=test.tgt
    :: wmt_src=@
    :: wmt_tgt=@
{
    data_dir="${wmt}/wmt23.${wmt_src}-${wmt_tgt}"
    cp "${data_dir}/test_corpus.src" $test_src
    cp "${data_dir}/test_corpus.ref" $test_tgt

    jq -r .src "${data_dir}/dev.jsonl" > $dev_src
    jq -r .ref "${data_dir}/dev.jsonl" > $dev_tgt
}

task BuildWMTInstructions
    < dev_src=@UnpackWMTData
    < dev_tgt=@UnpackWMTData
    < test_src=@UnpackWMTData
    > instructions=instructions.json
    :: template=@
    :: n_shots=@
    :: wmt_src=@
    :: wmt_tgt=@
    :: src_name=@
    :: tgt_name=@
    :: text2text_tokenizer=@
    :: repo=@
{

    # template, language pair, number of shots
    python $repo/scripts/build_instructions.py \
        --src $test_src \
        --shot-src $dev_src \
        --shot-tgt $dev_tgt \
        --src-lang $src_name \
        --tgt-lang $tgt_name \
        --template $template \
        --n-shots $n_shots \
        --chat-tokenizer $text2text_tokenizer \
        --out $instructions
}

task TranslateWMT
    < instructions=@BuildWMTInstructions
    > hyps=hyps.txt
    :: text2text_model=@
    :: text2text_tokenizer=@
    :: max_tokens=@
    :: repo=@
{

    python $repo/scripts/inference.py \
        --inpaths $instructions \
        --outpaths $hyps \
        --model $text2text_model \
        --tokenizer $text2text_tokenizer \
        --max-length $max_tokens

}

task TranslateWMTSeamless
    < test_src=@UnpackWMTData
    > hyps=hyps.txt
    :: text2text_model=@
    :: text2text_tokenizer=@
    :: flores_src=@
    :: seamless_src=@
    :: seamless_tgt=@
    :: seamless_batch_size=@
    :: max_tokens=@
    :: repo=@
{

    python $repo/scripts/inference.py \
        --inpaths $test_src \
        --outpaths $hyps \
        --model $text2text_model \
        --tokenizer $text2text_tokenizer \
        --max-length $max_tokens \
        --src-lang $seamless_src \
        --tgt-lang $seamless_tgt \
        --input-format raw_text \
        --batch-size $seamless_batch_size \
        --model-type encdec-speech \
        --backend hf \
        --seamless-input-type text

}

task ComputeWMTComet
    < hyps=@TranslateWMT
    < src=$test_src@UnpackWMTData
    < refs=$test_tgt@UnpackWMTData
    > comet=comet.json
    > means=comet-means.json
    :: comet_model=@
    :: repo=@
{
    # compute comet (we can get the other metrics in other tasks?)

    comet-score -s $src -t $hyps -r $refs --model $comet_model --to_json $comet

    python $repo/scripts/mean_comet.py --path $comet --mean-results $means
}

task ComputeWMTSPBleu
    < hyps=@TranslateWMT
    < refs=$test_tgt@UnpackWMTData
    > bleu=bleu.json
{

    sacrebleu --metrics bleu chrf --tokenize=flores200 $refs < $hyps > $bleu
}

task ComputeWMTCometSeamless
    < hyps=@TranslateWMTSeamless
    < src=$test_src@UnpackWMTData
    < refs=$test_tgt@UnpackWMTData
    > comet=comet.json
    > means=comet-means.json
    :: comet_model=@
    :: repo=@
{
    # compute comet (we can get the other metrics in other tasks?)

    comet-score -s $src -t $hyps -r $refs --model $comet_model --to_json $comet

    python $repo/scripts/mean_comet.py --path $comet --mean-results $means
}

task ComputeWMTSPBleuSeamless
    < hyps=@TranslateWMTSeamless
    < refs=$test_tgt@UnpackWMTData
    > bleu=bleu.json
{

    sacrebleu --metrics bleu chrf --tokenize=flores200 $refs < $hyps > $bleu
}

##### 5-shot Translation #####

plan TranslateFiveShot {
    reach TranslateWMT via (NShots: five) * (LanguagePair: en_de de_en en_ru ru_en en_zh zh_en) * (Text2TextModel: tower_base spire_base)
}

plan EvaluateFiveShot {
    reach ComputeWMTComet, ComputeWMTSPBleu via (NShots: five) * (LanguagePair: en_de de_en en_ru ru_en en_zh zh_en) * (Text2TextModel: tower_base spire_base)
}

##### 0-shot Translation #####

plan TranslateZeroShot {
    reach TranslateWMT via (NShots: zero) * (LanguagePair: en_de de_en en_ru ru_en en_zh zh_en) * (Text2TextModel: tower_instruct spire_full spire_no_blocks spire_no_pseudo tower_full)
}

plan EvaluateZeroShot {
    reach ComputeWMTComet, ComputeWMTSPBleu via (NShots: zero) * (LanguagePair: en_de de_en en_ru ru_en en_zh zh_en) * (Text2TextModel: tower_instruct spire_full spire_no_blocks spire_no_pseudo tower_full)
}

##### Seamless Translation #####

plan TranslateSeamless {
    reach TranslateWMTSeamless via (LanguagePair: en_de de_en en_ru ru_en en_zh zh_en) * (Text2TextModel: seamless)
}

plan EvaluateSeamless {
    reach ComputeWMTCometSeamless, ComputeWMTSPBleuSeamless via (LanguagePair: en_de de_en en_ru ru_en en_zh zh_en) * (Text2TextModel: seamless)
}
