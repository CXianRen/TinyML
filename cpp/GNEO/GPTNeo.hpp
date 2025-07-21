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
    MACRO_REGISTER_M_MEMEBR(attention_);
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
    MACRO_REGISTER_M_MEMEBR(c_fc_);
    MACRO_REGISTER_M_MEMEBR(c_proj_);
    MACRO_REGISTER_M_MEMEBR(act_);
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
  GPTNeoAttention<T> attention_;
  MLayerNorm<T> ln_2_;
  GPTNeoMLP<T> mlp_;

public:
  GPTNeoBlock(size_t embed_dim, size_t layer_id = 0)
      : layer_id_(layer_id),
      ln_1_(embed_dim, 1e-5),
      attention_(embed_dim, layer_id),
      ln_2_(embed_dim, 1e-5),
      mlp_(embed_dim, embed_dim * 4) {

    MACRO_CLASS_NAME(GPTNeoBlock);
    MACRO_REGISTER_M_MEMEBR(ln_1_);
    MACRO_REGISTER_M_MEMEBR(attention_);
    MACRO_REGISTER_M_MEMEBR(ln_2_);
    MACRO_REGISTER_M_MEMEBR(mlp_);
  }
        
  Tensor<T> forward(const Tensor<T>& hidden_states,
  									mtb::Tensor<T>* k_history=nullptr, 
										mtb::Tensor<T>* v_history=nullptr) {
    auto residual = hidden_states.copy();
    auto x = ln_1_.forward(hidden_states);
    x = attention_.forward(x, k_history, v_history);
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
    attention_.init_att(k, k_size, v, v_size, 
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
  std::vector<std::unique_ptr<GPTNeoBlock<T>>> layers_;
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
      layers_.emplace_back(std::make_unique<GPTNeoBlock<T>>(embed_dim, i));
    }

    MACRO_CLASS_NAME(GPTNeoModel);
    MACRO_REGISTER_M_MEMEBR(wte_);
    MACRO_REGISTER_M_MEMEBR(wpe_);
    // Register first layer as representative
    MACRO_REGISTER_M_MEMEBR((*(layers_[0].get()))); 
    MACRO_REGISTER_M_MEMEBR(ln_f_);
    MACRO_REGISTER_M_MEMEBR(lm_head_);
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
    // Load model weights from binary files
    
    auto wte_data = 
    load_data<T>(folder_path + 
      "/transformer/wte/weight/parameters.bin", 
                 vocab_size_ * embed_dim_, true);
    wte_.fill_weight(wte_data.data(), wte_data.size());

    auto wpe_data = 
    load_data<T>(folder_path + 
      "/transformer/wpe/weight/parameters.bin",
                  2048 * embed_dim_, true);
    wpe_.fill_weight(wpe_data.data(), wpe_data.size());

    // initialize layers
    for (size_t i = 0; i < num_layers_; ++i) {
      auto k_data = 
      load_data<T>(folder_path + 
        "/transformer/h/" + std::to_string(i) + 
        "/attn/attention/k_proj/weight/parameters.bin", 
                   embed_dim_ * embed_dim_, true);
      auto v_data = 
      load_data<T>(folder_path + 
        "/transformer/h/" + std::to_string(i) + 
        "/attn/attention/v_proj/weight/parameters.bin", 
                    embed_dim_ * embed_dim_, true);
      auto q_data = 
      load_data<T>(folder_path + 
        "/transformer/h/" + std::to_string(i) + 
        "/attn/attention/q_proj/weight/parameters.bin",
                   embed_dim_ * embed_dim_, true);
      auto o_w_data = 
      load_data<T>(folder_path + 
        "/transformer/h/" + std::to_string(i) + 
        "/attn/attention/out_proj/weight/parameters.bin",
                    embed_dim_ * embed_dim_, true);
      auto o_b_data = 
      load_data<T>(folder_path + 
        "/transformer/h/" + std::to_string(i) + 
        "/attn/attention/out_proj/bias/parameters.bin",
                   embed_dim_, true);
      
      // initialize attention
      layers_[i]->init_att(k_data.data(), k_data.size(),
                          v_data.data(), v_data.size(),
                          q_data.data(), q_data.size(),
                          o_w_data.data(), o_w_data.size(),
                          o_b_data.data(), o_b_data.size()); 
      
      // mlp weights
      auto c_fc_w_data = 
      load_data<T>(folder_path + "/transformer/h/" + std::to_string(i) + 
                   "/mlp/c_fc/weight/parameters.bin",
                   embed_dim_ * (embed_dim_ * 4), true);
    
      auto c_fc_b_data = 
      load_data<T>(folder_path + "/transformer/h/" + std::to_string(i) + 
                   "/mlp/c_fc/bias/parameters.bin",
                   embed_dim_ * 4, true);
      
      auto c_proj_w_data = 
      load_data<T>(folder_path + "/transformer/h/" + std::to_string(i) + 
                   "/mlp/c_proj/weight/parameters.bin",\
                    (embed_dim_ * 4) * embed_dim_, true);
      auto c_proj_b_data = 
      load_data<T>(folder_path + "/transformer/h/" + std::to_string(i) + 
                   "/mlp/c_proj/bias/parameters.bin",
                   embed_dim_, true);
      // initialize mlp
      layers_[i]->init_mlp(c_fc_w_data.data(), c_fc_w_data.size(),
                          c_fc_b_data.data(), c_fc_b_data.size(),
                          c_proj_w_data.data(), c_proj_w_data.size(),
                          c_proj_b_data.data(), c_proj_b_data.size());
      
      
      // layer norm weights
      auto ln_1_w_data = 
      load_data<T>(folder_path + "/transformer/h/" + std::to_string(i) + 
                   "/ln_1/weight/parameters.bin", embed_dim_, true);
      auto ln_1_b_data = 
      load_data<T>(folder_path + "/transformer/h/" + std::to_string(i) + 
                   "/ln_1/bias/parameters.bin", embed_dim_, true);

      auto ln_2_w_data = 
      load_data<T>(folder_path + "/transformer/h/" + std::to_string(i) + 
                   "/ln_2/weight/parameters.bin", embed_dim_, true);
      auto ln_2_b_data = 
      load_data<T>(folder_path + "/transformer/h/" + std::to_string(i) + 
                   "/ln_2/bias/parameters.bin", embed_dim_, true); 
      // initialize layer norm
      layers_[i]->init_ln_1(ln_1_w_data.data(), ln_1_w_data.size(),
                           ln_1_b_data.data(), ln_1_b_data.size());
      layers_[i]->init_ln_2(ln_2_w_data.data(), ln_2_w_data.size(),
                           ln_2_b_data.data(), ln_2_b_data.size()); 
    }
    // final layer norm
    auto ln_f_w_data = 
    load_data<T>(folder_path + 
      "/transformer/ln_f/weight/parameters.bin", embed_dim_, true);
    auto ln_f_b_data = 
    load_data<T>(folder_path + 
      "/transformer/ln_f/bias/parameters.bin", embed_dim_, true);
    ln_f_.fill_gamma(ln_f_w_data.data(), ln_f_w_data.size());
    ln_f_.fill_beta(ln_f_b_data.data(), ln_f_b_data.size()); 

    // lm head weights
    auto lm_head_w_data = 
    load_data<T>(folder_path + 
      "/lm_head/weight/parameters.bin", 
                 embed_dim_ * vocab_size_, true);
    lm_head_.fill_weight(lm_head_w_data.data(), lm_head_w_data.size());
  }
  
};