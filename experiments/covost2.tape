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

task GetCovostTrain
    > src_text=train.src
    > ref=train.ref
    > audio=audio.txt
    :: commonvoice_dir=@
    :: covost_src=@
    :: covost_tgt=@
{
    # this one doesn't call get_covost_splits.py, but rather assumes that they
    # already exist.

    covost_tsv="${commonvoice_dir}/covost_v2.${covost_src}_${covost_tgt}.train.tsv"
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

task ConvertTrainAudio
    < audio=@GetCovostTrain
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

task LabelTrainAudio
    < wav_tsv=@ConvertTrainAudio
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
        --validate-examples \
        --dtype $hubert_dtype
}

##### Direct ST tasks #####

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

##### ASR and cascaded ST tasks #####

task BuildASRInstructions
    < test_dsus=$dsus@LabelAudio
    > instructions=instructions.json
    :: speech2text_tokenizer=@
    :: repo=@
{
    # template, language pair, number of shots
    python $repo/scripts/build_instructions.py \
        --src $test_dsus \
        --template asr_simple \
        --n-shots 0 \
        --chat-tokenizer $speech2text_tokenizer \
        --out $instructions
}

task Transcribe
    < instructions=@BuildASRInstructions
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

# since it's a self-cascade, we'll just use the speech2text tokenizer
task BuildSelfCascadedInstructions
    < transcriptions=$hyps@Transcribe
    > instructions=instructions.json
    :: src_name=@
    :: tgt_name=@
    :: speech2text_tokenizer=@
    :: repo=@
{
    # template, language pair, number of shots
    python $repo/scripts/build_instructions.py \
        --src $transcriptions \
        --template mt_zero \
        --n-shots 0 \
        --src-lang $src_name \
        --tgt-lang $tgt_name \
        --chat-tokenizer $speech2text_tokenizer \
        --out $instructions
}

task DecodeCascaded
    < instructions=@BuildSelfCascadedInstructions
    > hyps=hyps.txt
    :: speech2text_model=@
    :: speech2text_tokenizer=@
    :: max_tokens=@
    :: repo=@
{

    python $repo/scripts/inference.py \
        --inpaths $instructions \
        --outpaths $hyps \
        --model $speech2text_model \
        --tokenizer $speech2text_tokenizer \
        --max-length $max_tokens

}

task CometCascaded
    < hyps=@DecodeCascaded
    < src=$src_text@GetAudioPathsAndCorpus
    < references=$ref@GetAudioPathsAndCorpus
    > comet=comet.json
    > means=comet-means.json
    :: comet_model=@
    :: repo=@
{
    # compute comet (we can get the other metrics in other tasks?)

    comet-score -s $src -t $hyps -r $references --model $comet_model --to_json $comet

    python $repo/scripts/mean_comet.py --path $comet --mean-results $means
}

task BleuCascaded
    < hyps=@DecodeCascaded
    < references=$ref@GetAudioPathsAndCorpus
    > bleu=bleu.json
    :: repo=@
{
    sacrebleu --metrics bleu chrf --tokenize=flores200 $references < $hyps > $bleu
}

task NormalizeHyps
    < hyps=@Transcribe
    > hyps_normalized=hyps.txt.norm
    :: normalizer=@
    :: repo=@
{
    python $repo/scripts/normalize.py --normalizer $normalizer < $hyps > $hyps_normalized
}

task NormalizeRefs
    < references=$src_text@GetAudioPathsAndCorpus
    > refs_normalized=refs.txt.norm
    :: normalizer=@
    :: repo=@
{
    python $repo/scripts/normalize.py --normalizer $normalizer < $references > $refs_normalized
}

task ComputeWER
    < hyps_normalized=@NormalizeHyps
    < refs_normalized=@NormalizeRefs
    > results=wer.json
    :: repo=@
{
    python $repo/scripts/compute_wer.py --hyp $hyps_normalized --ref $refs_normalized > $results
}

##### Plans for getting DSUs of the train set (which is parallel between en-de and en-zh) #####

plan GetTrain {
    reach GetCovostTrain via (LanguagePair: en_de en_zh)
}

plan ConvertTrain {
    reach ConvertTrainAudio
}

plan LabelizeTrain {
    reach LabelTrainAudio
}

##### Tokenize CoVoST2 #####

plan Labelize {
    reach LabelAudio via (LanguagePair: en_de en_zh)
}

##### Direct ST plans #####

plan DirectInstructions {
    reach BuildDirectInstructions via (Speech2TextModel: *) * (LanguagePair: en_de en_zh)
}

plan DirectTranslation {
    reach DecodeDirect via (Speech2TextModel: *) * (LanguagePair: en_de en_zh)
}

plan Evaluation {
    reach CometDirect, BleuDirect via (Speech2TextModel: *) * (LanguagePair: en_de en_zh)
}

plan EvaluationLMT {
    reach CometDirect, BleuDirect via (Speech2TextModel: lmt_0_6B_v2_all lmt_1_7B_v2_all_5200 lmt_4B_v2_all_2400 lmt_4B_v2_all_4400 lmt_1_7B_v2_all lmt_4B_v2_all_5600 lmt_4B_v2_all_8400 lmt_8B_v2_all_2800 lmt_4B_v2_all lmt_8B_v2_all_6400 lmt_4B_exdedup lmt_4B_ddb lmt_4B_no_asr_ddb lmt_4B_4lang lmt_4B_no_asr_4lang eurollm_9B_v2_all) * (LanguagePair: en_de en_zh)
}

##### Cascaded ST and ASR plans #####

plan TranscriptionInstructions {
    reach BuildASRInstructions via (Speech2TextModel: *) * (LanguagePair: en_de en_zh)
}

plan ASR {
    reach Transcribe via (Speech2TextModel: *) * (LanguagePair: en_de en_zh)
}

plan CascadedEvaluation {
    reach CometCascaded, BleuCascaded via (Speech2TextModel: *) * (LanguagePair: en_de en_zh)
}
