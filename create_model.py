"""
This script loads the specified GPT-2 model from OpenAI using TensorFlow,
converts it into our custom format and saves it to `model.gguf`, which contains
everything (all the parameters, all the weights, encoding/decoding
information).

Parts of this script were taken from the picoGPT project: https://github.com/jaymody/picoGPT

Those are licensed as:

MIT License

Copyright (c) 2023 Jay Mody

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

from time import monotonic as clock
import os
import json
import re
from shutil import copyfile

import numpy as np
import gguf
import requests
import tensorflow as tf
from tqdm import tqdm

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


def load_encoder_hparams_and_params(model_size, models_dir):
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

def convert(params, n_head, n_ctx, idx, decoder_txt,
        vocab_idx, vocab_txt, byte_decoder):
    t1 = clock()
    blocks = params["blocks"]
    n_embd = blocks[0]["ln_1"]["b"].size
    n_layer = len(blocks)
    mlp_fc_w = np.empty((n_layer,n_embd,4*n_embd), dtype=np.float32)
    mlp_fc_b = np.empty((n_layer,4*n_embd), dtype=np.float32)
    mlp_proj_w = np.empty((n_layer,4*n_embd,n_embd), dtype=np.float32)
    mlp_proj_b = np.empty((n_layer,n_embd), dtype=np.float32)
    attn_w = np.empty((n_layer,n_embd,3*n_embd), dtype=np.float32)
    attn_b = np.empty((n_layer,3*n_embd), dtype=np.float32)
    attn_proj_w = np.empty((n_layer,n_embd,n_embd), dtype=np.float32)
    attn_proj_b = np.empty((n_layer,n_embd), dtype=np.float32)
    ln1_g = np.empty((n_layer,n_embd), dtype=np.float32)
    ln1_b = np.empty((n_layer,n_embd), dtype=np.float32)
    ln2_g = np.empty((n_layer,n_embd), dtype=np.float32)
    ln2_b = np.empty((n_layer,n_embd), dtype=np.float32)
    for i, block in enumerate(blocks):
        mlp_fc_w[i,:,:] = block["mlp"]["c_fc"]["w"]
        mlp_fc_b[i,:] = block["mlp"]["c_fc"]["b"]
        mlp_proj_w[i,:,:] = block["mlp"]["c_proj"]["w"]
        mlp_proj_b[i,:] = block["mlp"]["c_proj"]["b"]
        attn_w[i,:,:] = block["attn"]["c_attn"]["w"]
        attn_b[i,:] = block["attn"]["c_attn"]["b"]
        attn_proj_w[i,:,:] = block["attn"]["c_proj"]["w"]
        attn_proj_b[i,:] = block["attn"]["c_proj"]["b"]
        ln1_g[i,:] = block["ln_1"]["g"]
        ln1_b[i,:] = block["ln_1"]["b"]
        ln2_g[i,:] = block["ln_2"]["g"]
        ln2_b[i,:] = block["ln_2"]["b"]
    wte = params["wte"]
    wpe = params["wpe"]
    lnf_g = params["ln_f"]["g"]
    lnf_b = params["ln_f"]["b"]
    t2 = clock()
    print("Transform time: ", t2-t1)
    t1 = clock()

    n_vocab = np.size(wte, 0)
    assert np.size(wte, 1) == n_embd

    model_type = 0xfa51697 # fastGPT
    model_version = 2
    header = np.array([model_type, model_version, n_vocab, n_ctx, n_embd, n_layer, n_head,
        len(idx),len(decoder_txt.encode("utf-8")),
        len(vocab_idx),len(vocab_txt.encode("utf-8")),len(byte_decoder)], dtype=np.int32)

    # Save the model to GGUF
    def save_gguf(data_offset_name, data_offset_value):
        g = gguf.GGUFWriter("model.gguf", None)
        g.add_int32(data_offset_name, data_offset_value)
        g.add_tensor("header", header)
        g.add_tensor("wte", wte); g.add_tensor("wpe", wpe)
        g.add_tensor("mlp_fc_w", mlp_fc_w); g.add_tensor("mlp_fc_b", mlp_fc_b)
        g.add_tensor("mlp_proj_w", mlp_proj_w); g.add_tensor("mlp_proj_b", mlp_proj_b)
        g.add_tensor("attn_w", attn_w); g.add_tensor("attn_b", attn_b)
        g.add_tensor("attn_proj_w", attn_proj_w); g.add_tensor("attn_proj_b",
                attn_proj_b)
        g.add_tensor("ln1_b", ln1_b); g.add_tensor("ln1_g", ln1_g)
        g.add_tensor("ln2_b", ln2_b); g.add_tensor("ln2_g", ln2_g)
        g.add_tensor("lnf_b", lnf_b); g.add_tensor("lnf_g", lnf_g)
        g.add_tensor("idx", idx)
        g.add_tensor("decoder_txt", np.frombuffer(decoder_txt.encode("utf-8"),
            dtype=np.int8))
        g.add_tensor("vocab_idx", vocab_idx)
        g.add_tensor("vocab_txt", np.frombuffer(vocab_txt.encode("utf-8"),
            dtype=np.int8))
        g.add_tensor("byte_decoder", byte_decoder)
        g.write_header_to_file()
        g.write_kv_data_to_file()
        g.write_tensors_to_file()
        g.close()

    data_offset_name = "general.data_offset"
    save_gguf(data_offset_name, 0)

    g = gguf.GGUFReader("model.gguf")
    data_offset = g.tensors[0].data_offset
    # * .offset: the offset of the kv entry
    # * 8: The i64 length of the key string
    # * 4: The i32 type of the value
    assert g.fields[data_offset_name].offset == 24
    offset_offset = g.fields[data_offset_name].offset + 8 + \
        len(data_offset_name) + 4
    print("offset offset:", offset_offset)
    print("data offset:", data_offset)

    save_gguf(data_offset_name, data_offset)

    t2 = clock()
    print("Save time: ", t2-t1)


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

def decoder_idx(decoder):
    i = 0
    idx = np.empty(len(decoder)+1, dtype=np.int32)
    idx[0] = i
    for n, t in enumerate(decoder):
        i += len(t.encode("utf-8"))
        idx[n+1] = i
    assert idx[-1] == len("".join(decoder).encode("utf-8"))
    return idx

def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    btu = dict(zip(bs, cs))
    byte_decoder = {v: k for k, v in btu.items()}
    bd = np.zeros(324, dtype=np.int32)
    for y in byte_decoder:
        x = ord(y)
        bd[x] = byte_decoder[y]
    bd2 = np.zeros(256, dtype=np.int32)
    for i in range(np.size(bd)):
        bd2[bd[i]] = i
    return bd2

def main(model_size: str = "124M", models_dir: str = "models"):
    print("Loading model")
    # load encoder, hparams, and params from the released open-ai gpt-2 files
    t1 = clock()
    hparams, params = load_encoder_hparams_and_params(model_size, models_dir)
    decoder = load_decoder(os.path.join(models_dir, model_size, "encoder.json"))
    vocab = load_vocab(os.path.join(models_dir, model_size, "vocab.bpe"))
    t2 = clock()
    print("  Done. Loading time: ", t2-t1)

    # generate output ids
    print("Converting model, saving to `model.gguf`")
    t1 = clock()
    decoder_txt = "".join(decoder)
    idx = decoder_idx(decoder)
    vocab_txt = "".join(vocab)
    vocab_idx = decoder_idx(vocab)
    byte_decoder = bytes_to_unicode()
    convert(params, hparams["n_head"], hparams["n_ctx"], idx, decoder_txt,
            vocab_idx, vocab_txt, byte_decoder)
    t2 = clock()
    print("  Done. Time: ", t2-t1)


if __name__ == "__main__":
    import fire
    fire.Fire(main)
