task ExtendTokenizer
    > tokenizer_base
    > tokenizer_instruct
    :: n_dsus=@
    :: base_model=@
    :: repo=@
{
    # challenge: in what order do we add stuff? The <CLS>, etc. needs to be
    # part of the sentencepiece vocab for everything to work out correctly

    # pre-dsu specials (like <CLS>) -- these need to be added to the spm model
    # post-dsu typs (like im_end) (are these ones actually specials?) -- shouldn't need to be added to the spm model

    python $repo/scripts/extend_spm.py \
        --original $base_model \
        --n_new_dsus $n_dsus \
        --new_specials "<CLS>,<EOD>,<MASK>,<PAD>,<SEP>" \
        --spm_prefix tmp \
        --hf_base $tokenizer_base \
        --hf_instruct $tokenizer_instruct
}



task ExtendModel
    < tokenizer=$tokenizer_base@ExtendTokenizer
    > extended_model=extended_model
    :: base_model=@
    :: init_strategy=@
    :: repo=@
{

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--model-path")
    # parser.add_argument("--out-dir")
    # parser.add_argument("--new-tokenizer", help="path to spm model")
    # parser.add_argument("--init-strategy", default="default", choices=["default", "mean"])
    # parser.add_argument("--pad-multiple", type=int, default=64)
    python $repo/scripts/extend_model.py \
        --model-path $base_model \
        --init-strategy $init_strategy \
        --new-tokenizer $tokenizer \
        --pad-multiple 64 \
        --out-dir $extended_model
}

plan PuaTokenizers {
    reach ExtendTokenizer
}

plan Extend {
    reach ExtendModel via (BaseModel: *)
}
