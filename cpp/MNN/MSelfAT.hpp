#ifndef __MSelfAT_HPP__
#define __MSelfAT_HPP__
#include "mtb.hpp"

#include "MLinear.hpp"

namespace mnn {

template <typename T>
class MSelfAT: public MModel {
private:
    size_t embed_size_;
    size_t num_attention_heads_;
    size_t head_dim_;

	MLinear<T> k_proj_;
	MLinear<T> v_proj_;
	MLinear<T> q_proj_;
	MLinear<T> out_proj_;

	mtb::Tensor<uint8_t> mask_;

public:
	MSelfAT(size_t embed_size, 
			size_t num_attention_heads, 
			size_t head_dim)
		: embed_size_(embed_size),
		  num_attention_heads_(num_attention_heads),
		  head_dim_(head_dim),
		  k_proj_(embed_size_, embed_size_, false),
		  v_proj_(embed_size_, embed_size_, false),
		  q_proj_(embed_size_, embed_size_, false),
		  out_proj_(embed_size_, embed_size_),
			mask_(mtb::triu<uint8_t>(
							mtb::ones<uint8_t>(
								{256, 256}), 1)) {
			MACRO_CLASS_NAME(MSelfAT);
			MACRO_REGISTER_M_MEMBER(k_proj_);
			MACRO_REGISTER_M_MEMBER(v_proj_);
			MACRO_REGISTER_M_MEMBER(q_proj_);
			MACRO_REGISTER_M_MEMBER(out_proj_);
		}
	
	mtb::Tensor<T> split_head(mtb::Tensor<T>& input) {
		// split the input tensor into multiple heads
		// [B, S, E] -> [B, S, H, D]
		auto reshaped = input.reshape(
				{input.shape()[0], input.shape()[1], 
				num_attention_heads_, head_dim_});
		// transpose to [B, H, S, D]
		return mtb::transpose(reshaped, {0, 2, 1, 3});
	}

	mtb::Tensor<T> forward(const mtb::Tensor<T>& input,
												  mtb::Tensor<T>* k_history=nullptr, 
													mtb::Tensor<T>* v_history=nullptr) 
		{
		// project input to query, key, and value
		// intput [B, S, E] -> q, k, v [B, S, E]
		auto q = q_proj_.forward(input);
		auto k = k_proj_.forward(input);
		auto v = v_proj_.forward(input);

		// split the tensors into multiple heads
		// [B, S, E] -> [B, H, S, D]
		q = split_head(q);
		k = split_head(k);
		v = split_head(v);

		if (k_history != nullptr && v_history != nullptr) {
			// if k and v history are provided, append the new k and v
			// [B, H, S, D] -> [B, H, S + S', D]
			if (k_history->shape().size() != 1 && 
					v_history->shape().size() != 1) {
					k = mtb::concatenate(std::vector<mtb::Tensor<T>>({*k_history, k}), 2);
					v = mtb::concatenate(std::vector<mtb::Tensor<T>>({*v_history, v}), 2);
			}
			// update the history tensors
			*k_history = k;
			*v_history = v;
		}

		// calculate attention scores
		// [B, H, S, D] * [B, H, D, S] -> [B, H, S, S]
		auto scores = mtb::matmul(q, mtb::transpose(k, {0, 1, 3, 2}));

		// apply mask
		size_t q_len = q.shape()[2];
		size_t k_len = k.shape()[2];
	  // maks[k_len-q_len:k_len, :k_len] 
		// only when the k_ken-q_len > 1, the mask is required
		auto mask = mask_.slice({{k_len-q_len, k_len}, {0, k_len}});
		auto scores_masked = mtb::where(mask, T(-INFINITY), scores);

		// softmax along the last dimension
		auto weights = mtb::softmax(scores_masked, -1); 

		// calculate attention output
		// [B, H, S, D]
		auto attn_output = mtb::matmul(weights, v); 
		
		// transpose back to  [B, S, H, D]
		// copy is making sure we have a contiguous tensor
		attn_output = mtb::transpose(attn_output, {0, 2, 1, 3}).copy(); 
	  
		// reshape to [B, S, H * D]
		attn_output = attn_output.reshape(
				{attn_output.shape()[0], attn_output.shape()[1], 
				num_attention_heads_ * head_dim_});

		// project again to output
		// [B, S, E]
		attn_output = out_proj_.forward(attn_output); 
		return attn_output;

	}

	void fill_k(T* data, size_t size) {
		k_proj_.fill_weight(data, size);
	}

	void fill_v(T* data, size_t size) {
		v_proj_.fill_weight(data, size);
	}

	void fill_q(T* data, size_t size) {
		q_proj_.fill_weight(data, size);
	}

	void fill_out(T* weights, T* bias, size_t w_size, size_t b_size) {
		out_proj_.fill_weight(weights, w_size);
		out_proj_.fill_bias(bias, b_size);
	}

};

} // namespace mnn

#endif // __MSelfAT_HPP__
