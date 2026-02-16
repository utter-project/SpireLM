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
    < devtest=@FetchFlores
    > instructions=instructions.json
    :: template=@
    :: n_shots=@
    :: src_name=@
    :: tgt_name=@
    :: text2text_tokenizer=@
    :: repo=@
{
    python $repo/scripts/build_instructions.py \
        --src $devtest/"devtest.${flores_src}" \
        --src-names example src_lang tgt_lang \
        --src-constants $src_name $tgt_name \
        --templates $template \
        --template-key mt \
        --tokenizer $text2text_tokenizer \
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
