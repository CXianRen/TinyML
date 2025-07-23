# TinyML, a MiniGPT  — Pure C++ Implementation

This project provides a minimal implementation of a GPT-like model using **pure C++**, reproducing the structure and inference of the [TinyStories-33M](https://huggingface.co/roneneldan/TinyStories-33M) model.

It is organized into two main parts:

---

## 🔹 Python Prototype (`python/`)

A reference implementation using **NumPy**, designed to help verify the model structure and logic before translating to C++.  
Includes:
- A simple transformer model built with NumPy
- A minimal Byte-Pair Encoding (BPE) tokenizer

---

## 🔹 C++ Core Implementation (`cpp/`)

The C++ version contains three main modules:

### 1. **MTB** (Mini Tensor Backend)  
A lightweight tensor library that supports:
- Tensors with arbitrary dimensions
- Auto broadcasting
- Basic operations required for Transformer models
- numpy style api

### 2. **MNN** (Mini Neural Network)  
A neural network framework built on MTB, implementing:
- Embedding layer  
- Linear (fully connected) layer  
- Layer Normalization  
- Activation functions (e.g. GELU)  
- Self-Attention layer  

### 3. **GNEO**  
Implements the GPT-Neo style blocks used in TinyStories-33M with cachecing machanism.
Also includes a simple tokenizer for mapping between characters and embedding IDs.

---

## 🔻 Download Pretrained Weights

Download the pre-converted model parameters from the original TinyStories-33M checkpoint:

👉 [Download Link](https://drive.google.com/file/d/1r_Kf6FWjWpf49-N1624788A9wjFU5fjH/view?usp=sharing)

Unzip the downloaded file into the root directory. You should get the following folder: "model_state_dic/"


## 🚀 Usage

### ▶️ Run the C++ Version

```bash
cd cpp
mkdir build
make run
```

You should see output like the following:

![demo](./doc/imgs/demo_with_cache.gif)

### ▶️ Run the Python Version
```bash
python python/test_model_v2.py
```

## Profiler
```sh
  GPTNeoModel::forward: 15531 us
    Embedding Layer: 10 us
    Transformer Layers: 9109 us
      GPTNeoBlock::forward: 2186 us
        Attention Layer: 794 us
      GPTNeoBlock::forward: 2349 us
        Attention Layer: 993 us
      GPTNeoBlock::forward: 2330 us
        Attention Layer: 936 us
      GPTNeoBlock::forward: 2233 us
        Attention Layer: 877 us
    LN&LM: 6411 us
```

## Structure printer
```sh
 GPTNeoModel(
  (wte) : MEmbed(50257, 768)
  (wpe) : MEmbed(2048, 768)
  (layers) : MList size:4 (
    (0) : GPTNeoBlock(
      (ln_1) : MLayerNorm(768, 1e-05)
      (attn) : GPTNeoAttention(
        (attention) : MSelfAT(
          (k_proj) : MLinear (in_features: 768, out_features: 768, has_bias: false)
          (v_proj) : MLinear (in_features: 768, out_features: 768, has_bias: false)
          (q_proj) : MLinear (in_features: 768, out_features: 768, has_bias: false)
          (out_proj) : MLinear (in_features: 768, out_features: 768, has_bias: true)
        )
      )
      (ln_2) : MLayerNorm(768, 1e-05)
      (mlp) : GPTNeoMLP(
        (c_fc) : MLinear (in_features: 768, out_features: 3072, has_bias: true)
        (c_proj) : MLinear (in_features: 3072, out_features: 768, has_bias: true)
        (act) :MGELUActivation
      )
    )
  )
  (ln_f) : MLayerNorm(768, 1e-05)
  (lm_head) : MLinear (in_features: 768, out_features: 50257, has_bias: false)
)
```

## TODO
- Add configuration