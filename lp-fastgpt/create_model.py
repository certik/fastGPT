"""This script loads the specified GPT-2 model from OpenAI
using TensorFlow, converts it into our custom format and saves
it to `model.dat`, which contains everything (all the
parameters, all the weights, encoding/decoding information).

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
import os
import json
import re


import numpy as np
import requests
import tensorflow as tf
from tqdm import tqdm


from typing import Optional, Union, Any


def download_gpt2_files(model_size, model_dir):
    assert model_size in ["124M", "355M", "774M", "1558M"]
    for filename in [
        "checkpoint",
        "encoder.json",
        "hparams.json",
        "model.ckpt.data-00000-of-00001",
        "model.ckpt.index",
        "model.ckpt.meta",
        "vocab.bpe",
    ]:
        url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
        r = requests.get(f"{url}/{model_size}/{filename}", stream=True)
        r.raise_for_status()

        with open(os.path.join(model_dir, filename), "wb") as f:
            file_size = int(r.headers["content-length"])
            chunk_size = 1000
            with tqdm(
                    ncols=100,
                    desc="Fetching " + filename,
                    total=file_size,
                    unit_scale=True,
            ) as pbar:
                # 1k for chunk_size, since Ethernet packet size is around 1500 bytes
                for chunk in r.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
                    pbar.update(chunk_size)


def load_gpt2_params_from_tf_ckpt(tf_ckpt_path, hparams):
    def set_in_nested_dict(d, keys, val):
        if not keys:
            return val
        if keys[0] not in d:
            d[keys[0]] = {}
        d[keys[0]] = set_in_nested_dict(d[keys[0]], keys[1:], val)
        return d

    init_vars = tf.train.list_variables(tf_ckpt_path)
    params = {"blocks": [{} for _ in range(hparams["n_layer"])]}
    for name, _ in init_vars:
        array = np.squeeze(tf.train.load_variable(tf_ckpt_path, name))
        name = name.removeprefix("model/")
        if name.startswith("h"):
            m = re.match(r"h([0-9]+)/(.*)", name)
            n = int(m[1])
            sub_name = m[2]
            set_in_nested_dict(params["blocks"][n], sub_name.split("/"), array)
        else:
            set_in_nested_dict(params, name.split("/"), array)

    return params


def tf_load_encoder_hparams_and_params(model_size, models_dir):

    assert model_size in ["124M", "355M", "774M", "1558M"]

    model_dir = os.path.join(models_dir, model_size)

    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)

    if not tf_ckpt_path:  # download files if necessary
        os.makedirs(model_dir, exist_ok=True)
        download_gpt2_files(model_size, model_dir)
        tf_ckpt_path = tf.train.latest_checkpoint(model_dir)

    hparams = json.load(open(os.path.join(model_dir, "hparams.json")))

    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, hparams)

    return hparams, params


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


def convert(params,
            n_head,
            n_ctx,
            decoder_idx,
            decoder_txt,
            vocab_idx,
            vocab_txt,
            byte_decoder) -> Model:

    t1 = clock()

    # must predefine just to get shapes ...
    blocks : ParamsBlocksType = params["blocks"]
    nblocks = len(blocks)
    assert nblocks == NBlocks

    n_embd  = blocks[0]["ln_1"]["b"].size
    n_layer = nblocks
    assert n_layer == NBlocks

    n_vocab       = ParamsWteShape[0]  # np.size(mo.wte, 0)
    model_type    = ModelType
    model_version = ModelVersion

    mo : Model = make_empty_model_with_metadata(
        model_type       = model_type,
        model_version    = model_version,
        n_vocab          = n_vocab,
        n_ctx            = n_ctx,
        n_embd           = n_embd,
        n_layer          = n_layer,
        n_head           = n_head,
        decoder_idx_len  = DecoderIdxShape[0],
        # It's useful to check magic numbers against computations:
        decoder_txt_len  = len(decoder_txt.encode("utf-8")),
        vocab_idx_len    = VocabIdxShape[0],
        vocab_txt_len    = len(vocab_txt.encode("utf-8")),
        byte_decoder_len = len(byte_decoder),
    )

    for i, block in enumerate(blocks):
        mo.mlp_fc_w[i, :, :]    = block["mlp"]["c_fc"]["w"]
        mo.mlp_fc_b[i, :]       = block["mlp"]["c_fc"]["b"]
        mo.mlp_proj_w[i, :, :]  = block["mlp"]["c_proj"]["w"]
        mo.mlp_proj_b[i, :]     = block["mlp"]["c_proj"]["b"]
        mo.attn_w[i, :, :]      = block["attn"]["c_attn"]["w"]
        mo.attn_b[i, :]         = block["attn"]["c_attn"]["b"]
        mo.attn_proj_w[i, :, :] = block["attn"]["c_proj"]["w"]
        mo.attn_proj_b[i, :]    = block["attn"]["c_proj"]["b"]

        mo.ln1_g[i, :]          = block["ln_1"]["g"]
        mo.ln1_b[i, :]          = block["ln_1"]["b"]
        mo.ln2_g[i, :]          = block["ln_2"]["g"]
        mo.ln2_b[i, :]          = block["ln_2"]["b"]

    mo.wte   = params["wte"]
    mo.wpe   = params["wpe"]

    mo.lnf_g = params["ln_f"]["g"]
    mo.lnf_b = params["ln_f"]["b"]

    mo.decoder_idx  = decoder_idx
    mo.decoder_txt  = decoder_txt
    mo.vocab_idx    = vocab_idx
    mo.vocab_txt    = vocab_txt
    mo.byte_decoder = byte_decoder

    t2 = clock()
    print("Transform time: ", t2 - t1)

    check_model(mo)

    # Save the model
    t1 = clock()
    save_model(mo)
    t2 = clock()
    print("Save time: ", t2 - t1)

    # Round-trip the model: check the stored version.

    t1 = clock()
    m = restore_model()

    check_model(m)
    assert np.all(m.wte          == mo.wte)
    assert np.all(m.wpe          == mo.wpe)
    assert np.all(m.mlp_fc_w     == mo.mlp_fc_w)
    assert np.all(m.mlp_fc_b     == mo.mlp_fc_b)
    assert np.all(m.mlp_proj_w   == mo.mlp_proj_w)
    assert np.all(m.mlp_proj_b   == mo.mlp_proj_b)
    assert np.all(m.attn_w       == mo.attn_w)
    assert np.all(m.attn_b       == mo.attn_b)
    assert np.all(m.attn_proj_w  == mo.attn_proj_w)
    assert np.all(m.attn_proj_b  == mo.attn_proj_b)
    assert np.all(m.ln1_b        == mo.ln1_b)
    assert np.all(m.ln1_g        == mo.ln1_g)
    assert np.all(m.ln2_b        == mo.ln2_b)
    assert np.all(m.ln2_g        == mo.ln2_g)
    assert np.all(m.lnf_b        == mo.lnf_b)
    assert np.all(m.lnf_g        == mo.lnf_g)
    assert np.all(m.decoder_idx  == mo.decoder_idx)
    assert m.decoder_txt         == mo.decoder_txt
    assert np.all(m.vocab_idx    == mo.vocab_idx)
    assert m.vocab_txt           == mo.vocab_txt
    assert np.all(m.byte_decoder == mo.byte_decoder)
    t2 = clock()
    print("Restore time: ", t2 - t1)

    return m


def save_model(mo : Model):
    with open("model.dat", "w") as f:
        model_metadata = np.array(
            [
                mo.model_type,
                mo.model_version,
                mo.n_vocab,
                mo.n_ctx,
                mo.n_embd,
                mo.n_layer,
                mo.n_head,
                len(mo.decoder_idx),
                len(mo.decoder_txt.encode("utf-8")),
                len(mo.vocab_idx),
                len(mo.vocab_txt.encode("utf-8")),
                len(mo.byte_decoder)],
            dtype=np.int32)

        assert model_metadata.shape == ModelMetadataShape

        model_metadata.tofile(f)

        mo.wte.tofile(f)
        mo.wpe.tofile(f)

        mo.mlp_fc_w.tofile(f)
        mo.mlp_fc_b.tofile(f)

        mo.mlp_proj_w.tofile(f)
        mo.mlp_proj_b.tofile(f)

        mo.attn_w.tofile(f)
        mo.attn_b.tofile(f)

        mo.attn_proj_w.tofile(f)
        mo.attn_proj_b.tofile(f)

        mo.ln1_b.tofile(f)
        mo.ln1_g.tofile(f)
        mo.ln2_b.tofile(f)
        mo.ln2_g.tofile(f)

        mo.lnf_b.tofile(f)
        mo.lnf_g.tofile(f)

        mo.decoder_idx.tofile(f)
        f.write(mo.decoder_txt)
        mo.vocab_idx.tofile(f)
        f.write(mo.vocab_txt)
        mo.byte_decoder.tofile(f)


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


def load_decoder(filename):
    D = json.load(open(filename))
    D2 = {v: k for k, v in D.items()}
    i = 0
    decoder = []
    while True:
        if i not in D2:
            break
        decoder.append(D2[i])
        i += 1
    return decoder


def load_vocab(filename):
    D = open(filename).read()
    D = D.split("\n")
    D = D[1:]
    return D


def load_decoder_idx(decoder):
    i = 0
    idx = np.empty(len(decoder) + 1, dtype=np.int32)
    idx[0] = i
    for n, t in enumerate(decoder):
        i += len(t.encode("utf-8"))
        idx[n + 1] = i
    assert idx[-1] == len("".join(decoder).encode("utf-8"))
    return idx


def bytes_to_unicode() -> np.ndarray:
    bs = list(range(ord("!"), ord("~") + 1)) + \
        list(range(ord("¡"), ord("¬") + 1)) + \
        list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    btu = dict(zip(bs, cs))
    byte_decoder = {v: k for k, v in btu.items()}
    bd = np.zeros(324, dtype=np.int32)
    for y in byte_decoder:
        x = ord(y)
        bd[x] = byte_decoder[y]
    bd2 = np.zeros(DecoderShape, dtype=np.int32)
    for i in range(np.size(bd)):
        bd2[bd[i]] = i
    return bd2


def main(model_size: str = "124M",
         models_dir: str = "models") -> Model:

    # ================================================================
    # load encoder, hparams, and params from the released open-ai gpt-2 files
    print("Loading model")
    t1 = clock()

    hparams : HParamsType
    params  : ParamsType

    hparams, params = \
        tf_load_encoder_hparams_and_params(model_size, models_dir)

    decoder : list[str] = \
        load_decoder(os.path.join(models_dir, model_size, "encoder.json"))

    assert len(decoder) == NVocab  # TODO: ??? !!! ???

    vocab : list[str] = \
        load_vocab(os.path.join(models_dir, model_size, "vocab.bpe"))

    assert len(vocab) == 50_001

    t2 = clock()
    print("  Done. Loading time: ", t2 - t1)
    # ================================================================
    # generate output ids
    print("Converting model, saving to `model.dat`")
    t1 = clock()

    decoder_idx : np.ndarray = load_decoder_idx(decoder)
    assert decoder_idx.shape == DecoderIdxShape

    decoder_txt : str = "".join(decoder)
    assert len(decoder_txt) == DecoderTxtAsciiLen

    vocab_idx    = load_decoder_idx(vocab)
    assert vocab_idx.shape == VocabIdxShape

    vocab_txt    = "".join(vocab)
    assert len(vocab_txt) == VocabTxtAsciiLen

    byte_decoder = bytes_to_unicode()
    assert byte_decoder.shape == DecoderShape

    m : Model = \
        convert(params,
                hparams["n_head"],
                hparams["n_ctx"],
                decoder_idx,
                decoder_txt,
                vocab_idx,
                vocab_txt,
                byte_decoder)

    t2 = clock()
    print("  Done. Time: ", t2 - t1)
    # ================================================================

    return m


if __name__ == "__main__":
    import fire

    fire.Fire(main)
