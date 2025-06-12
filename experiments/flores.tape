# - download Flores
task FetchFlores
    > dev=dev
    > devtest=devtest
    :: flores_path=@
    :: repo=@
{
    python $repo/scripts/hf2text.py --dev $dev --devtest $devtest
}

task BuildFloresInstructions
    < dev=@FetchFlores
    < devtest=@FetchFlores
    > instructions=instructions.json
    :: template=@
    :: n_shots=@
    :: flores_src=@
    :: flores_tgt=@
    :: src_name=@
    :: tgt_name=@
    :: text2text_tokenizer=@
    :: repo=@
{

    # template, language pair, number of shots
    python $repo/scripts/build_instructions.py \
        --src $devtest/"devtest.${flores_src}" \
        --shot-src "${dev}/dev.${flores_src}" \
        --shot-tgt "${dev}/dev.${flores_tgt}" \
        --src-lang $src_name \
        --tgt-lang $tgt_name \
        --template $template \
        --n-shots $n_shots \
        --chat-tokenizer $text2text_tokenizer \
        --out $instructions
}

task TranslateFlores
    < instructions=@BuildFloresInstructions
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

task TranslateFloresSeamless
    < devtest=@FetchFlores
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
        --inpaths "${devtest}/devtest.${flores_src}" \
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

task ComputeFloresComet
    < hyps=@TranslateFlores
    < devtest=@FetchFlores
    > comet=comet.json
    > means=comet-means.json
    :: flores_src=@
    :: flores_tgt=@
    :: comet_model=@
    :: repo=@
{
    SRC="${devtest}/devtest.${flores_src}"
    REF="${devtest}/devtest.${flores_tgt}"

    comet-score -s $SRC -t $hyps -r $REF --model $comet_model --to_json $comet

    python $repo/scripts/mean_comet.py --path $comet --mean-results $means
}

task ComputeFloresSPBleu
    < hyps=@TranslateFlores
    < devtest=@FetchFlores
    > bleu=bleu.json
    :: flores_src=@
    :: flores_tgt=@
    :: comet_model=@
    :: repo=@
{

    REF="${devtest}/devtest.${flores_tgt}"
    sacrebleu --metrics bleu chrf --tokenize=flores200 $REF < $hyps > $bleu
}

task ComputeFloresCometSeamless
    < hyps=@TranslateFloresSeamless
    < devtest=@FetchFlores
    > comet=comet.json
    > means=comet-means.json
    :: flores_src=@
    :: flores_tgt=@
    :: comet_model=@
    :: repo=@
{
    # compute comet (we can get the other metrics in other tasks?)

    SRC="${devtest}/devtest.${flores_src}"
    REF="${devtest}/devtest.${flores_tgt}"

    comet-score -s $SRC -t $hyps -r $REF --model $comet_model --to_json $comet

    python $repo/scripts/mean_comet.py --path $comet --mean-results $means
}

task ComputeFloresSPBleuSeamless
    < hyps=@TranslateFloresSeamless
    < devtest=@FetchFlores
    > bleu=bleu.json
    :: flores_src=@
    :: flores_tgt=@
    :: comet_model=@
    :: repo=@
{

    REF="${devtest}/devtest.${flores_tgt}"
    sacrebleu --metrics bleu chrf --tokenize=flores200 $REF < $hyps > $bleu
}

##### 5-shot Translation #####

plan TranslateFiveShot {
    reach TranslateFlores via (NShots: five) * (LanguagePair: *) * (Text2TextModel: tower_base spire_base)
}

plan EvaluateFiveShot {
    reach ComputeFloresComet, ComputeFloresSPBleu via (NShots: five) * (LanguagePair: *) * (Text2TextModel: tower_base spire_base)
}

##### 0-shot Translation #####

plan TranslateZeroShot {
    reach TranslateFlores via (NShots: zero) * (LanguagePair: *) * (Text2TextModel: tower_instruct spire_full spire_no_blocks spire_no_pseudo tower_full)
}

plan EvaluateZeroShot {
    reach ComputeFloresComet, ComputeFloresSPBleu via (NShots: zero) * (LanguagePair: *) * (Text2TextModel: tower_instruct spire_full spire_no_blocks spire_no_pseudo tower_full)
}

plan TranslateZeroShotAll {
    reach TranslateFlores via (NShots: zero) * (LanguagePair: *) * (Text2TextModel: *)
}

plan EvaluateZeroShotAll {
    reach ComputeFloresComet, ComputeFloresSPBleu via (NShots: zero) * (LanguagePair: *) * (Text2TextModel: *)
}

##### Seamless Translation #####

plan TranslateFloresSeamless {
    reach TranslateFloresSeamless via (NShots: zero) * (LanguagePair: *) * (Text2TextModel: seamless)
}

plan EvaluateFloresSeamless {
    reach ComputeFloresCometSeamless, ComputeFloresSPBleuSeamless via (NShots: zero) * (LanguagePair: *) * (Text2TextModel: seamless)
}
