task ExtendTokenizer
    > tokenizer_base
    > tokenizer_instruct
    :: n_dsus=@
    :: tower_base_spm=@
    :: repo=@
{
    # challenge: in what order do we add stuff? The <CLS>, etc. needs to be
    # part of the sentencepiece vocab for everything to work out correctly

    # pre-dsu specials (like <CLS>) -- these need to be added to the spm model
    # post-dsu typs (like im_end) (are these ones actually specials?) -- shouldn't need to be added to the spm model

    python $repo/scripts/extend-spm.py \
        --original $tower_base_spm \
        --n_new_dsus $n_dsus \
        --new_specials "<CLS>,<EOD>,<MASK>,<PAD>,<SEP>" \
        --spm_prefix extended \
        --hf_base $tokenizer_base \
        --hf_instruct $tokenizer_instruct
}

task FormatFleursPUA
    > dsus=dsu.txt
    :: fleurs_dsus=@
    :: repo=@
{
    python $repo/tokenizer-scripts/format_dsus.py --format pua < $fleurs_dsus > $dsus
}

task BuildFleursInstructions
    < tokenizer_instruct=@ConvertTokenizer
    < test_dsus=$dsus@FormatFleursPUA
    > instructions=instructions.json
    :: inference_repo=@
{
    # template, language pair, number of shots
    python $inference_repo/generation-scripts/build_instructions.py \
        --src $test_dsus \
        --template asr_simple \
        --n-shots 0 \
        --chat-tokenizer $tokenizer_instruct \
        --out $instructions
}

task Decode
    < instructions=@BuildFleursInstructions
    < tokenizer_instruct=@ConvertTokenizer
    > hyps=hyps.txt
    :: spire_full=@
    :: inference_repo=@
{
    python $inference_repo/generation-scripts/inference.py \
        --inpaths $instructions \
        --outpaths $hyps \
        --model $spire_full \
        --tokenizer $tokenizer_instruct \
        --backend vllm \
        --max-length 256

}

task DecodeSlow
    < instructions=@BuildFleursInstructions
    < tokenizer_instruct=@ConvertTokenizer
    > hyps=hyps.txt
    :: spire_full=@
    :: inference_repo=@
{
    python $inference_repo/generation-scripts/inference.py \
        --inpaths $instructions \
        --outpaths $hyps \
        --model $spire_full \
        --tokenizer $tokenizer_instruct \
        --backend vllm \
        --slow-tokenizer \
        --max-length 256

}

plan PuaTokenizers {
    reach ConvertTokenizer
}

plan BuildInstructions {
    reach BuildFleursInstructions
}

plan TestPuaDecode {
    reach Decode
}

plan TestPuaDecodeSlow {
    reach DecodeSlow
}
