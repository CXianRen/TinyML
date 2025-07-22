#pragma once
#include "mnn.hpp"
#include <memory>

using namespace mtb;
using namespace mnn;

template<typename T>
class GPTNeoAttention: public MModel {
private:
  size_t layer_id_;
  size_t embed_dim_; 
  MSelfAT<T> attention_;

public: 
  GPTNeoAttention(size_t embed_dim = 768, size_t layer_id = 0)
      : layer_id_(layer_id), 
      embed_dim_(embed_dim),
      attention_(embed_dim, 16, embed_dim/16) {
    MACRO_CLASS_NAME(GPTNeoAttention);
    MACRO_REGISTER_M_MEMBER(attention_);
  }

  Tensor<T> forward(const Tensor<T>& hidden_states,
                    mtb::Tensor<T>* k_history = nullptr, 
                    mtb::Tensor<T>* v_history = nullptr) {
    return attention_.forward(hidden_states, k_history, v_history);
  }

  void init_att(T* k, size_t k_size,
                T* v, size_t v_size,
                T* q, size_t q_size,
                T* o_w, size_t o_w_size,
                T* o_b, size_t o_b_size) {
    attention_.fill_k(k, k_size);
    attention_.fill_v(v, v_size);
    attention_.fill_q(q, q_size);
    attention_.fill_out(o_w, o_b, o_w_size, o_b_size);          
  }
};

template<typename T>
class GPTNeoMLP: public MModel {
private:
  size_t embed_dim_;
  size_t inermediate_dim_;
  MLinear<T> c_fc_;
  MLinear<T> c_proj_;
  MGELUActivation<T> act_;

public:
  GPTNeoMLP(size_t embed_dim, size_t inermediate_dim)
      : embed_dim_(embed_dim),
      inermediate_dim_(inermediate_dim),
      c_fc_(embed_dim, inermediate_dim, true),
      c_proj_(inermediate_dim, embed_dim, true),
      act_() {
    MACRO_CLASS_NAME(GPTNeoMLP);
    MACRO_REGISTER_M_MEMBER(c_fc_);
    MACRO_REGISTER_M_MEMBER(c_proj_);
    MACRO_REGISTER_M_MEMBER(act_);
  }

  Tensor<T> forward(const Tensor<T>& hidden_states) {
    auto x = c_fc_.forward(hidden_states);
    x = act_.forward(x);
    x = c_proj_.forward(x);
    return x;
  }

  void init_mlp(T* c_fc_w, size_t c_fc_w_size,
                T* c_fc_b, size_t c_fc_b_size,
                T* c_proj_w, size_t c_proj_w_size,
                T* c_proj_b, size_t c_proj_b_size) {
    c_fc_.fill_weight(c_fc_w, c_fc_w_size);
    c_fc_.fill_bias(c_fc_b, c_fc_b_size);
    c_proj_.fill_weight(c_proj_w, c_proj_w_size);
    c_proj_.fill_bias(c_proj_b, c_proj_b_size);
  }
};


template<typename T>
class GPTNeoBlock: public MModel {
private:
  size_t layer_id_;
  MLayerNorm<T> ln_1_;
  GPTNeoAttention<T> attn_;
  MLayerNorm<T> ln_2_;
  GPTNeoMLP<T> mlp_;

public:
  GPTNeoBlock(size_t embed_dim, size_t layer_id = 0)
      : layer_id_(layer_id),
      ln_1_(embed_dim, 1e-5),
      attn_(embed_dim, layer_id),
      ln_2_(embed_dim, 1e-5),
      mlp_(embed_dim, embed_dim * 4) {

    MACRO_CLASS_NAME(GPTNeoBlock);
    MACRO_REGISTER_M_MEMBER(ln_1_);
    MACRO_REGISTER_M_MEMBER(attn_);
    MACRO_REGISTER_M_MEMBER(ln_2_);
    MACRO_REGISTER_M_MEMBER(mlp_);
  }
        
  Tensor<T> forward(const Tensor<T>& hidden_states,
  									mtb::Tensor<T>* k_history=nullptr, 
										mtb::Tensor<T>* v_history=nullptr) {
    auto residual = hidden_states.copy();
    auto x = ln_1_.forward(hidden_states);
    x = attn_.forward(x, k_history, v_history);
    x += residual; // Add residual connection
    
    residual = x.copy();
    x = ln_2_.forward(x);
    x = mlp_.forward(x);
    x += residual; // Add residual connection
    return x;
  }
  
  void init_att(T* k, size_t k_size,
                T* v, size_t v_size,
                T* q, size_t q_size,
                T* o_w, size_t o_w_size,
                T* o_b, size_t o_b_size) {
    attn_.init_att(k, k_size, v, v_size, 
      q, q_size, o_w, o_w_size, o_b, o_b_size);
  }

  void init_mlp(T* c_fc_w, size_t c_fc_w_size,
                T* c_fc_b, size_t c_fc_b_size,
                T* c_proj_w, size_t c_proj_w_size,
                T* c_proj_b, size_t c_proj_b_size) {
    mlp_.init_mlp(c_fc_w, c_fc_w_size, 
                  c_fc_b, c_fc_b_size,
                  c_proj_w, c_proj_w_size,
                  c_proj_b, c_proj_b_size);
  }

  void init_ln_1(T* ln_1_w, size_t ln_1_w_size,
                 T* ln_1_b, size_t ln_1_b_size) {
    ln_1_.fill_gamma(ln_1_w, ln_1_w_size);
    ln_1_.fill_beta(ln_1_b, ln_1_b_size);
  }

  void init_ln_2(T* ln_2_w, size_t ln_2_w_size,
                 T* ln_2_b, size_t ln_2_b_size) {
    ln_2_.fill_gamma(ln_2_w, ln_2_w_size);
    ln_2_.fill_beta(ln_2_b, ln_2_b_size);
  }
};

template<typename T>
class GPTNeoModel: public MModel {
private:
  size_t num_layers_;
  size_t embed_dim_;
  size_t vocab_size_;
  MEmbed<T> wte_;
  MEmbed<T> wpe_;
  MList<GPTNeoBlock<T>> layers_;
  MLayerNorm<T> ln_f_;
  MLinear<T> lm_head_;
public:
  GPTNeoModel(size_t num_layers, 
              size_t embed_dim, 
              size_t vocab_size=50257):
        num_layers_(num_layers),
        embed_dim_(embed_dim),
        vocab_size_(vocab_size),
        wte_(vocab_size, embed_dim),
        wpe_(2048, embed_dim),
        layers_(), // Initialize layers as an empty vector
        ln_f_(embed_dim, 1e-5),
        lm_head_(embed_dim, vocab_size, false) {
    for (size_t i = 0; i < 4; ++i) {
      layers_.add(std::make_unique<GPTNeoBlock<T>>(embed_dim, i));
    }

    MACRO_CLASS_NAME(GPTNeoModel);
    MACRO_REGISTER_M_MEMBER(wte_);
    MACRO_REGISTER_M_MEMBER(wpe_);
    // Register first layer as representative
    MACRO_REGISTER_M_MEMBER(layers_);
    MACRO_REGISTER_M_MEMBER(ln_f_);
    MACRO_REGISTER_M_MEMBER(lm_head_);
    printInfo();
  }

  Tensor<T> forward(const Tensor<int>& input_ids, 
                    const Tensor<int>& position_ids,
                    std::vector<Tensor<T>>* k_history = nullptr,
                    std::vector<Tensor<T>>* v_history = nullptr) {
    auto hidden_states = wte_.forward(input_ids);
    auto position_embeds = wpe_.forward(position_ids);

    hidden_states += position_embeds; // Add position embeddings
    for (size_t i = 0; i < num_layers_; ++i) {
      if (k_history != nullptr && v_history != nullptr) 
      {
        // If k_history and v_history are provided, pass them to the attention layer
        hidden_states = layers_[i]->forward(hidden_states, 
                                           &((*k_history)[i]), 
                                           &((*v_history)[i]));
      } else {
        // Otherwise, just forward the hidden states
        hidden_states = layers_[i]->forward(hidden_states);
      }
    }
    
    hidden_states = ln_f_.forward(hidden_states);
    auto logits = lm_head_.forward(hidden_states);
    return logits; // Return logits for next token prediction
  }

  void load_model(const std::string& folder_path) {
    wte_.loadParameters(folder_path + "/transformer/wte/");
    wpe_.loadParameters(folder_path + "/transformer/wpe/");

    // initialize layers
    layers_.loadParameters(folder_path + "/transformer/h/");
   
    ln_f_.loadParameters(folder_path + "/transformer/ln_f/");
    lm_head_.loadParameters(folder_path + "/lm_head/");
  }
};
