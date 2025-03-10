from itertools import count, repeat

from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
import sentencepiece as spm

from transformers import LlamaTokenizer


TOWER_INSTRUCT_CHAT_TEMPLATE = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content']}}{% if (loop.last and add_generation_prompt) or not loop.last %}{{ '<|im_end|>' + '\n'}}{% endif %}{% endfor %}{% if add_generation_prompt and messages[-1]['role'] != 'assistant' %}{{ '<|im_start|>assistant\n' }}{% endif %}"


def _load_model(path):
    sp_model = spm.SentencePieceProcessor(model_file=path)
    proto = sp_pb2_model.ModelProto()
    proto.ParseFromString(sp_model.serialized_model_proto())
    return sp_model, proto


def _create_piece(string, score=0, special=False):
    new_piece = sp_pb2_model.ModelProto().SentencePiece()
    new_piece.piece = string
    new_piece.score = score

    if special:
        new_piece.type = 4  # will this line ever cause a problem?
    return new_piece


def _extend(proto, new_types, scoring, special=False):
    if scoring == "bpe":
        get_score = count(proto.pieces[-1].score - 1, -1)
    else:
        get_score = repeat(0)

    # extend the model proto
    for new_type in new_types:
        new_piece = _create_piece(new_type, next(get_score), special=special)
        proto.pieces.append(new_piece)

    # you can also update the vocab size in proto.trainer_spec, but it doesn't
    # seem to matter

    return proto


def _save(proto, model_prefix):
    with open(model_prefix + ".model", "wb") as f:
        f.write(proto.SerializeToString())

    with open(model_prefix + ".vocab", "w") as f:
        for piece in proto.pieces:
            f.write("\t".join([piece.piece, str(int(piece.score))]) + "\n")


def extend_spm(original_model, save_prefix, new_specials, new_types, scoring_type="bpe", new_types_are_special=False):
    original_hf = LlamaTokenizer.from_pretrained(original_model)
    model, proto = _load_model(original_hf.vocab_file)
    proto = _extend(proto, new_specials, scoring_type, special=True)
    proto = _extend(proto, new_types, scoring_type, special=new_types_are_special)
    _save(proto, save_prefix)


def convert_spm_to_hf(spm_prefix, use_eos_as_pad=True):
    hf_model = LlamaTokenizer(vocab_file=spm_prefix + ".model")
    if use_eos_as_pad:
        hf_model.pad_token_id = hf_model.eos_token_id
        hf_model.pad_token = hf_model.eos_token
    return hf_model


def add_instruct_extras(hf_model, im_start_is_special=True):
    hf_model.add_special_tokens(
        {"eos_token": "<|im_end|>"}
    )
    if im_start_is_special:
        hf_model.add_special_tokens({"additional_special_tokens": ["<|im_start|>"]})
    else:
        hf_model.add_tokens("<|im_start|>")
    hf_model.chat_template = TOWER_INSTRUCT_CHAT_TEMPLATE
    return hf_model
