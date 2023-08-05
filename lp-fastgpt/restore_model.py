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


from time import monotonic as clock
import numpy as np
from typing import Union

from dataclasses import dataclass


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
MlpCProjBShape     = (          NEmbed,)
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

HParamsType = dict[str, int]


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
    byte_decoder_len : int
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
    byte_decoder     : np.ndarray


def restore_model() -> Model:
    m : Model = make_empty_model_with_metadata(
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
        byte_decoder_len = 0,
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
    m.byte_decoder_len = metadata[11]

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

    floff, m.byte_decoder = restore_ints(DecoderShape, floff)

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


def make_empty_model_with_metadata(
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
        byte_decoder_len : int,) -> Model:

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
        byte_decoder_len = byte_decoder_len,

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
        byte_decoder = np.empty(0                             , dtype=np.int32),
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
    assert mo.byte_decoder.shape == (mo.byte_decoder_len,)


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
    assert mo.byte_decoder_len == DecoderShape[0]


def main() -> Model:

    print("Restoring model")
    t1 = clock()
    m : Model = restore_model()
    t2 = clock()
    print("  Done. Time: ", t2 - t1)
    # ================================================================

    return m


if __name__ == "__main__":
    import fire
    fire.Fire(main)
