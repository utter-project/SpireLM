# todo: make this about covost

task GetAudioPathsAndCorpus
    > src_text=test.src
    > ref=test.ref
    > audio=audio.txt
    :: commonvoice_dir=@
    :: covost_src=@
    :: covost_tgt=@
    :: repo=@
{
    # commonvoice_dir contains all of the tsv files (including the ones specific
    # to en-de and en-zh)
    python $repo/scripts/get_covost_splits.py \
        --version 2 --src-lang $covost_src --tgt-lang $covost_tgt \
        --root $commonvoice_dir --cv-tsv "${commonvoice_dir}/validated.tsv"

    covost_tsv="${commonvoice_dir}/covost_v2.${covost_src}_${covost_tgt}.test.tsv"
    cut -f 1 $covost_tsv | tail -n +2 > $audio
    cut -f 2 $covost_tsv | tail -n +2 > $src_text
    cut -f 3 $covost_tsv | tail -n +2 > $ref
}

# This should replace everything up to ConvertTSV, I think
task ConvertAudio
    < audio=@GetAudioPathsAndCorpus
    > wav_tsv=wav.tsv
    > wav
    :: commonvoice_dir=@
    :: repo=@
{
    python $repo/scripts/convert_covost.py \
        --audio-list $audio \
        --audio-dir $commonvoice_dir/clips \
        --output-dir $wav \
        --output-tsv $wav_tsv
}

task LabelAudio
    < wav_tsv=@ConvertAudio
    > dsus=dsu.txt
    :: hubert_model=@
    :: hubert_layer=@
    :: km_model=@
    :: hubert_dtype=@
    :: repo=@
{
    python $repo/scripts/label-audio.py \
        --tsv_path $wav_tsv \
        --out_path $dsus \
        --ckpt_path $hubert_model \
        --layer $hubert_layer \
        --km_path $km_model \
        --batch-size 1000000 \
        --dtype $hubert_dtype
}

task BuildDirectInstructions
    < test_dsus=$dsus@LabelAudio
    > instructions=instructions.json
    :: st_template=@
    :: speech2text_tokenizer=@
    :: tgt_name=@
    :: repo=@
{
    # template, language pair, number of shots
    python $repo/scripts/build_instructions.py \
        --src $test_dsus \
        --template $st_template \
        --n-shots 0 \
        --chat-tokenizer $speech2text_tokenizer \
        --tgt-lang $tgt_name \
        --out $instructions
}

task DecodeDirect
    < instructions=@BuildDirectInstructions
    > hyps=hyps.txt
    :: speech2text_model=@
    :: speech2text_tokenizer=@
    :: max_tokens=@
    :: backend=@
    :: repo=@
{
    python $repo/scripts/inference.py \
        --inpaths $instructions \
        --outpaths $hyps \
        --model $speech2text_model \
        --tokenizer $speech2text_tokenizer \
        --backend $backend \
        --max-length $max_tokens

}

task CometDirect
    < hyps=@DecodeDirect
    < src=$src_text@GetAudioPathsAndCorpus
    < references=$ref@GetAudioPathsAndCorpus
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
    < references=$ref@GetAudioPathsAndCorpus
    > bleu=bleu.json
    :: repo=@
{
    sacrebleu --metrics bleu chrf --tokenize=flores200 $references < $hyps > $bleu
}

##### Tokenize CoVoST2 #####

plan Labelize {
    reach LabelAudio via (LanguagePair: en_de en_zh)
}

##### Prep data #####

plan DirectInstructions {
    reach BuildDirectInstructions via (Speech2TextModel: *) * (LanguagePair: en_de en_zh)
}

##### Inference #####

plan DirectTranslation {
    reach DecodeDirect via (Speech2TextModel: *) * (LanguagePair: en_de en_zh)
}

##### Evaluation #####

plan Evaluation {
    reach CometDirect, BleuDirect via (Speech2TextModel: *) * (LanguagePair: en_de en_zh)
}
