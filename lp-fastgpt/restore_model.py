"""This script restores the model from `model.dat`, which
contains everything (all the parameters, all the weights,
encoding/decoding information).

Parts of this script were taken from the picoGPT project:
https://github.com/jaymody/picoGPT

Those are licensed as:

MIT License

Copyright (c) 2023 Jay Mody

Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the
Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY
KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""

import numpy as np
from typing import Union

from dataclasses import dataclass

from timer import Timer


# === Magic (Unexplained) Numbers =========================
DecoderIdxType     = np.ndarray
DecoderIdxShape    = (50_258,)

DecoderTxtType     = str
DecoderTxtAsciiLen = 320_827
DecoderTxtUtf8Len  = 356_735  # It's checked somewhere below.

VocabIdxType       = np.ndarray
VocabIdxShape      = (50_002,)

VocabTxtType       = str
VocabTxtAsciiLen   = 370_558
VocabTxtUtf8Len    = 406_304  # It's checked somewhere below.
NVocab             = 50_257

DecoderShape       = (256,)
DecoderLen         = DecoderShape[0]

NBlocks            = 12

ModelType          = 0xfa51697
ModelVersion       = 1

NCtx               = 1024
NEmbed             = 768

Magic2304          = 2304
Magic3072          = 3072

# === Block-Attention components ======================
CAttnBType         = np.ndarray
CAttnBShape        = (        Magic2304,)
CAttnWType         = np.ndarray
CAttnWShape        = (NEmbed, Magic2304,)
CAttnType          = dict[str, Union[CAttnBType, CAttnWType]]

CProjBType         = np.ndarray
CProjBShape        = (        NEmbed,)
CProjWType         = np.ndarray
CProjWShape        = (NEmbed, NEmbed,)
CProjType          = dict[str, Union[CProjBType, CProjWType]]
# --- top level
BlockAttnType      = dict[str, Union[CAttnType, CProjType]]

# === BlockLn components ==============================
BlockLnBType       = np.ndarray
BlockLnBShape      = (NEmbed,)
BlockLnGType       = np.ndarray
BlockLnGShape      = (NEmbed,)
# --- top level
BlockLnType        = dict[str, Union[BlockLnBType, BlockLnGType]]

# === BlockMlp components =============================
MlpCFcBType        = np.ndarray
MlpCFcBShape       = (        Magic3072,)
MlpCFcWType        = np.ndarray
MlpCFcWShape       = (NEmbed, Magic3072,)
MlpCFcType         = dict[str, Union[MlpCFcBType, MlpCFcWType]]

MlpCProjBType      = np.ndarray
MlpCProjBShape     = (           NEmbed,)
MlpCProjWType      = np.ndarray
MlpCProjWShape     = (Magic3072, NEmbed,)
MlpCProjType       = dict[str, Union[MlpCProjBType, MlpCProjWType]]
# --- top level
BlockMlpType       = dict[str, Union[MlpCFcType, MlpCProjType]]

# =====================================================
ParamsBlockType    = dict[str, Union[BlockAttnType,
                                     BlockLnType,  # two of these
                                     BlockMlpType]]

ParamsBlocksType   = list[ParamsBlockType]
ParamsLnFType      = dict[str, np.ndarray]
ParamsLnFValShape  = BlockLnBShape
ParamsWpeType      = np.ndarray
ParamsWpeShape     = (  NCtx, NEmbed,)
ParamsWteType      = np.ndarray
ParamsWteShape     = (NVocab, NEmbed,)
ParamsType         = dict[str, Union[ParamsBlocksType,
                                     ParamsLnFType,
                                     ParamsWpeType,
                                     ParamsWteType]]

ModelMetadataType  = np.ndarray
ModelMetadataShape = (12,)

NTokensToGenerate  =   20
MaxTokens          = 2048

@dataclass
class Model:
    # integer metadata
    model_type       : int
    model_version    : int
    n_vocab          : int
    n_ctx            : int
    n_embd           : int
    n_layer          : int
    n_head           : int
    decoder_idx_len  : int
    decoder_txt_len  : int
    vocab_idx_len    : int
    vocab_txt_len    : int
    byte_encoder_len : int
    # check shapes in asserts for now; type system too weak.
    mlp_fc_w         : np.ndarray
    mlp_fc_b         : np.ndarray
    mlp_proj_w       : np.ndarray
    mlp_proj_b       : np.ndarray
    attn_w           : np.ndarray
    attn_b           : np.ndarray
    attn_proj_w      : np.ndarray
    attn_proj_b      : np.ndarray
    ln1_g            : np.ndarray
    ln1_b            : np.ndarray
    ln2_g            : np.ndarray
    ln2_b            : np.ndarray
    wte              : np.ndarray
    wpe              : np.ndarray
    lnf_g            : np.ndarray
    lnf_b            : np.ndarray
    # auxiliary matrices and texts
    decoder_idx      : np.ndarray
    decoder_txt      : str
    vocab_idx        : np.ndarray
    vocab_txt        : str
    byte_encoder     : np.ndarray


def restore_model() -> Model:
    m : Model = empty_model_with_metadata(
        model_type       = 0,
        model_version    = 0,
        n_vocab          = 0,
        n_ctx            = 0,
        n_embd           = 0,
        n_layer          = 0,
        n_head           = 0,
        decoder_idx_len  = 0,
        decoder_txt_len  = 0,
        vocab_idx_len    = 0,
        vocab_txt_len    = 0,
        byte_encoder_len = 0,
        )

    floff : int = 0
    metadata : np.ndarray = (
        np.fromfile("model.dat",
                    dtype=np.int32,
                    count=ModelMetadataShape[0],
                    offset=floff))

    m.model_type       = metadata[ 0]
    m.model_version    = metadata[ 1]
    m.n_vocab          = metadata[ 2]
    m.n_ctx            = metadata[ 3]
    m.n_embd           = metadata[ 4]
    m.n_layer          = metadata[ 5]
    m.n_head           = metadata[ 6]
    m.decoder_idx_len  = metadata[ 7]
    m.decoder_txt_len  = metadata[ 8]
    m.vocab_idx_len    = metadata[ 9]
    m.vocab_txt_len    = metadata[10]
    m.byte_encoder_len = metadata[11]

    check_model_metadata(m)

    floff += ModelMetadataShape[0] * BYTES_PER_INT32
    floff, m.wte         = restore_floats(ParamsWteShape              , floff)
    floff, m.wpe         = restore_floats(ParamsWpeShape              , floff)
    floff, m.mlp_fc_w    = restore_floats((NBlocks,) + MlpCFcWShape   , floff)
    floff, m.mlp_fc_b    = restore_floats((NBlocks,) + MlpCFcBShape   , floff)
    floff, m.mlp_proj_w  = restore_floats((NBlocks,) + MlpCProjWShape , floff)
    floff, m.mlp_proj_b  = restore_floats((NBlocks,) + MlpCProjBShape , floff)
    floff, m.attn_w      = restore_floats((NBlocks,) + CAttnWShape    , floff)
    floff, m.attn_b      = restore_floats((NBlocks,) + CAttnBShape    , floff)
    floff, m.attn_proj_w = restore_floats((NBlocks,) + CProjWShape    , floff)
    floff, m.attn_proj_b = restore_floats((NBlocks,) + CProjBShape    , floff)
    floff, m.ln1_b       = restore_floats((NBlocks,) + BlockLnBShape  , floff)
    floff, m.ln1_g       = restore_floats((NBlocks,) + BlockLnBShape  , floff)
    floff, m.ln2_b       = restore_floats((NBlocks,) + BlockLnBShape  , floff)
    floff, m.ln2_g       = restore_floats((NBlocks,) + BlockLnBShape  , floff)
    floff, m.lnf_b       = restore_floats(ParamsLnFValShape           , floff)
    floff, m.lnf_g       = restore_floats(ParamsLnFValShape           , floff)
    floff, m.decoder_idx = restore_ints(DecoderIdxShape               , floff)

    with open("model.dat", "rb") as f:
        f.seek(floff)
        decoder_txti_ub : bytes = f.read(DecoderTxtUtf8Len)
        m.decoder_txt = decoder_txti_ub.decode("utf-8")
        assert len(m.decoder_txt) == DecoderTxtAsciiLen
        floff += DecoderTxtUtf8Len

    floff, m.vocab_idx = restore_ints(VocabIdxShape, floff)

    with open("model.dat", "rb") as f:
        f.seek(floff)
        vocab_txt_ub : bytes = f.read(VocabTxtUtf8Len)
        m.vocab_txt = vocab_txt_ub.decode("utf-8")
        assert len(m.vocab_txt) == VocabTxtAsciiLen
        floff += VocabTxtUtf8Len

    floff, m.byte_encoder = restore_ints(DecoderShape, floff)

    return m


def prod_tuple(t : tuple[int]) -> int:
    result : int = 1
    for e in t:
        result *= e
    return result


BYTES_PER_INT32   = 4
BYTES_PER_FLOAT32 = 4


def restore_floats(shape : tuple, offset : int) -> tuple[int, np.ndarray]:
    """agnostic to length of shape; TODO impossible to statically type"""
    result : np.ndarray
    count = prod_tuple(shape)
    result = np.fromfile("model.dat",
                         dtype=np.float32,
                         count=count,
                         offset=offset)
    result = np.reshape(result, shape)
    new_offset : int = offset + (count * BYTES_PER_FLOAT32)
    return new_offset, result


def restore_ints(shape : tuple, offset : int) -> tuple[int, np.ndarray]:
    """agnostic to length of shape; TODO impossible to statically type"""
    result : np.ndarray
    count = prod_tuple(shape)
    result = np.fromfile("model.dat",
                         dtype=np.int32,
                         count=count,
                         offset=offset)
    result = np.reshape(result, shape)
    new_offset : int = offset + (count * BYTES_PER_INT32)
    return new_offset, result


def empty_model_with_metadata(
        model_type       : int,
        model_version    : int,
        n_vocab          : int,
        n_ctx            : int,
        n_embd           : int,
        n_layer          : int,
        n_head           : int,
        decoder_idx_len  : int,
        decoder_txt_len  : int,
        vocab_idx_len    : int,
        vocab_txt_len    : int,
        byte_encoder_len : int,) -> Model:

    mo: Model = Model(
        model_type       = model_type,
        model_version    = model_version,
        n_vocab          = n_vocab,
        n_ctx            = n_ctx,
        n_embd           = n_embd,
        n_layer          = n_layer,
        n_head           = n_head,
        decoder_idx_len  = decoder_idx_len,
        decoder_txt_len  = decoder_txt_len,
        vocab_idx_len    = vocab_idx_len,
        vocab_txt_len    = vocab_txt_len,
        byte_encoder_len = byte_encoder_len,

        mlp_fc_w     = np.empty((n_layer, n_embd, 4 * n_embd) , dtype=np.float32),
        mlp_fc_b     = np.empty((n_layer, 4 * n_embd)         , dtype=np.float32),
        mlp_proj_w   = np.empty((n_layer, 4 * n_embd, n_embd) , dtype=np.float32),
        mlp_proj_b   = np.empty((n_layer, n_embd)             , dtype=np.float32),
        attn_w       = np.empty((n_layer, n_embd, 3 * n_embd) , dtype=np.float32),
        attn_b       = np.empty((n_layer, 3 * n_embd)         , dtype=np.float32),
        attn_proj_w  = np.empty((n_layer, n_embd, n_embd)     , dtype=np.float32),
        attn_proj_b  = np.empty((n_layer, n_embd)             , dtype=np.float32),
        ln1_g        = np.empty((n_layer, n_embd)             , dtype=np.float32),
        ln1_b        = np.empty((n_layer, n_embd)             , dtype=np.float32),
        ln2_g        = np.empty((n_layer, n_embd)             , dtype=np.float32),
        ln2_b        = np.empty((n_layer, n_embd)             , dtype=np.float32),
        wte          = np.empty(0                             , dtype=np.float32),
        wpe          = np.empty(0                             , dtype=np.float32),
        lnf_g        = np.empty(0                             , dtype=np.float32),
        lnf_b        = np.empty(0                             , dtype=np.float32),

        decoder_idx  = np.empty(0                             , dtype=np.int32),
        decoder_txt  = '',
        vocab_idx    = np.empty(0                             , dtype=np.int32),
        vocab_txt    = '',
        byte_encoder = np.empty(0                             , dtype=np.int32),
    )
    return mo


def check_model(mo : Model) -> None:
    check_model_metadata(mo)

    assert mo.mlp_fc_w.shape     == (NBlocks,) + MlpCFcWShape
    assert mo.mlp_fc_b.shape     == (NBlocks,) + MlpCFcBShape
    assert mo.mlp_proj_w.shape   == (NBlocks,) + MlpCProjWShape
    assert mo.mlp_proj_b.shape   == (NBlocks,) + MlpCProjBShape
    assert mo.attn_w.shape       == (NBlocks,) + CAttnWShape
    assert mo.attn_b.shape       == (NBlocks,) + CAttnBShape
    assert mo.attn_proj_w.shape  == (NBlocks,) + CProjWShape
    assert mo.attn_proj_b.shape  == (NBlocks,) + CProjBShape
    assert mo.ln1_g.shape        == (NBlocks,) + BlockLnGShape
    assert mo.ln1_b.shape        == (NBlocks,) + BlockLnBShape
    assert mo.ln2_g.shape        == (NBlocks,) + BlockLnGShape
    assert mo.ln2_b.shape        == (NBlocks,) + BlockLnBShape
    assert mo.wte.shape          == ParamsWteShape
    assert mo.wpe.shape          == ParamsWpeShape
    assert mo.lnf_g.shape        == BlockLnGShape
    assert mo.lnf_b.shape        == BlockLnBShape

    assert mo.n_vocab            == ParamsWteShape[0]
    assert np.size(mo.wte, 1)    == NEmbed
    assert mo.decoder_idx.shape  == (mo.decoder_idx_len,)
    assert mo.vocab_idx.shape    == (mo.vocab_idx_len,)
    assert mo.byte_encoder.shape == (mo.byte_encoder_len,)


def check_model_metadata(mo):
    assert mo.model_type       == ModelType
    assert mo.model_version    == ModelVersion
    assert mo.n_vocab          == NVocab
    assert mo.n_ctx            == NCtx
    assert mo.n_embd           == NEmbed
    assert mo.n_layer          == NBlocks
    assert mo.n_head           == NBlocks

    assert mo.decoder_idx_len  == DecoderIdxShape[0]
    assert mo.decoder_txt_len  == DecoderTxtUtf8Len
    assert mo.vocab_idx_len    == VocabIdxShape[0]
    assert mo.vocab_txt_len    == VocabTxtUtf8Len
    assert mo.byte_encoder_len == DecoderShape[0]


def tokenize_word(input_ : str, i : int) -> tuple[str, int]:
    result = ('', i)
    i0 : int = i
    if len(input_) >= 1 and input_[i0] == ' ':
        i += 1
    while i < len(input_):
        ci: str = input_[i]
        if ci == ' ' or ci == '.' or ci == ',':
            result = (input_[i0:i], i)
            return result
        i += 1
    result = (input_[i0:i], i)
    return result


def next_token(input_ : str, i : int) -> tuple[str, int]:
    result = ('', i)
    if i >= len(input_):
        return result
    ci : str = input_[i]
    if ci == ' ':
        result = tokenize_word(input_, i)
    elif ci == ',' or ci == '.':
        i += 1
        result = (ci, i)
    else:
        result = tokenize_word(input_, i)
    return result

def merge_pair(intokens : list[bytearray], idx : int) -> list[bytearray]:
    tokens       : list[bytearray]
    i: int
    tokens = intokens[:len(intokens)-1].copy()
    tokens[idx] = (intokens[idx].decode("utf-8") + intokens[idx + 1].decode("utf-8")).encode("utf-8")
    for i in range(idx+2, len(intokens)):
        tokens[i-1] = intokens[i]
    return tokens


def merge_utf8_pairs(tokens : list[bytearray]) -> list[bytearray]:
    i             : int
    j             : int
    one_more_pass : bool = True

    j = 0
    while one_more_pass and j<len(tokens):
        one_more_pass = False
        i = j
        while i < len(tokens):
            if len(tokens[i]) == 1 and tokens[i][0]>=128:
                tokens = merge_pair(tokens, i)
                one_more_pass = True
                j = i + 1
            i += 1
    return tokens


def word_idx(word: str, idx: np.ndarray, decoder_txt: str) -> int:
    for i in range(len(idx) - 1):
        if decoder_txt[idx[i]:idx[i + 1]].decode("utf-8") == word:
            return i
    return -1

def bpe(m : Model, token : str) -> list[bytearray]:
    tokens         : list[bytearray] = []
    not_found      : int =  0
    merge_pair_idx : int =  0
    i              : int =  0
    MAGIC          : int = 10
    x              : int

    not_found = len(m.vocab_idx) + MAGIC
    for t in token:
        tokens.append(t.encode("utf-8"))
    tokens = merge_utf8_pairs(tokens)
    # pair_scores => size of tokens-1
    vocab_txt = m.vocab_txt.encode("utf-8")
    while len(tokens) > 1:
        pair_scores    : list =  []
        for i in range(0, len(tokens) - 1):
            x = word_idx(tokens[i].decode("utf-8") + " " + tokens[i + 1].decode("utf-8"), m.vocab_idx, vocab_txt)
            if x != -1:
                pair_scores.append(x)
            else:
                pair_scores.append(not_found)
        merge_pair_idx = pair_scores.index(min(pair_scores))

        if pair_scores[merge_pair_idx] == not_found:
            break

        tokens = merge_pair(tokens, merge_pair_idx)
    return tokens


def utf8_to_codepoint(s: str, i: int) -> tuple[int, int]:
    c = ord(s[i])
    if c >= 128:
        i += 1
        d = ord(s[i])
        c = (((c&31)<<6)|(d&63))
    if c >= 2048:
        raise Exception("error in utf8 to codepoint")
    return (c, i)


def codepoint_to_utf8(s: str, c: int) -> str:
    if c < 128:
        return s + chr(c)
    elif c < 2048:
        d1: int = ((c>>6)|192)
        d2:int = ((c|128)&191)
        return s + chr(d1) + chr(d2)
    else:
        raise Exception("Error in codepoint_to_utf8")

def encode(m : Model, input_ : str, byte_decoder : np.ndarray) -> np.ndarray:
    """Compare to the fortran function in tokenizer.f90."""
    # reshape this later after counting tokens
    tokens2  : np.ndarray = np.zeros(MaxTokens, dtype=np.int32)
    n_tokens : int        = 0
    i        : int        = 0  # fortran counts from 1
    j : int = 0
    # Python does not have \p for punctuation.
    # rex = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    # rex.match(input_)
    decoder_txt = m.decoder_txt.encode("utf-8")
    while True:
        tmp : str
        tmp, i = next_token(input_, i)
        if len(tmp) == 0:
            break
        t : str
        tmp2: str = ""
        for t in tmp:
            c : int = ord(t)
            c = m.byte_encoder[c]
            tmp2 += chr(c)
        bpe_tokens: list[bytearray] = bpe(m, tmp2)
        for j in range(len(bpe_tokens)):
            n_tokens = n_tokens + 1
            if (n_tokens > MaxTokens):
                raise Exception("Error in encode")
            tokens2[n_tokens-1] = word_idx(bpe_tokens[j].decode("utf-8"), m.decoder_idx, decoder_txt)
    return tokens2[:n_tokens]

def decode(tokens, idx, decoder_txt, byte_decoder) -> str:
    output2: str = ""
    output: str = ""
    i: int
    for i in range(len(tokens)):
        if tokens[i] < 0:
            raise Exception("less than 0")
        output2 += decoder_txt[idx[tokens[i]]:idx[tokens[i]+1]].decode("utf-8")
    i = 0
    while i < len(output2):
        c = ord(output2[i])
        if c < 0 or c > len(byte_decoder):
            raise Exception("Codepoint out of range for byte decoder")
        tmp:str = chr(byte_decoder[c])
        output += tmp
        i = i + 1
    return output

def gpt2_driver(m : Model, input_ : str) -> str:
    print(f"Input to the model =\n{input_}")
    n_tokens_to_generate : int = NTokensToGenerate
    n_seq                : int = len(input_)
    byte_encoder_max     : int = np.max(m.byte_encoder)
    byte_decoder : np.ndarray = \
        np.zeros((byte_encoder_max + 1,), dtype=np.int32)
    for i, e in enumerate(m.byte_encoder):
        byte_decoder[m.byte_encoder[i]] = i
    encoded : np.ndarray = encode(m, input_, byte_decoder)
    print("Input tokens:\n", list(encoded))
    decoder_txt = m.decoder_txt.encode("utf-8")
    result: str = decode(encoded, m.decoder_idx, decoder_txt, byte_decoder)
    assert result == input_, "Encoder-Decoder failed"
    return result


def main() -> Model:

    with Timer(text="Restored the model in {:0.4f} seconds."):
        m : Model = restore_model()

    input : str = """Alan Turing theorized that computers would one day become very powerful, but even he could not imagine"""
    with Timer(text="Ran the model in {:0.6f} seconds."):
        result : str = gpt2_driver(m, input)
        print(result)
    return m


if __name__ == "__main__":
    main()
