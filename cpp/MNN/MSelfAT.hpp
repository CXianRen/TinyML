#ifndef __MSelfAT_HPP__
#define __MSelfAT_HPP__
#include "mtb.hpp"

#include "MLinear.hpp"

namespace mnn {

template <typename T>
class MSelfAT {
public:
	MSelfAT(int embed_size, 
			int num_attention_heads, 
			int head_dim)
		: embed_size_(embed_size),
		  num_attention_heads_(num_attention_heads),
		  head_dim_(head_dim),
		  k_proj_(embed_size_, embed_size_, false),
		  v_proj_(embed_size_, embed_size_, false),
		  q_proj_(embed_size_, embed_size_, false),
		  out_proj_(embed_size_, embed_size_)
		{}
		
	mtb::Tensor<T> forward(mtb::Tensor<T>& input) {
		// project input to query, key, and value
		// intput [B, S, E] -> q, k, v [B, S, E]
		auto q = q_proj_.forward(input);
		auto k = k_proj_.forward(input);
		auto v = v_proj_.forward(input);

		// split the tensors into multiple heads
		// [B, S, E] -> [B, S, H, D]
		q = q.reshape(
				{q.shape()[0], q.shape()[1], 
				num_attention_heads_, head_dim_});
		q = mtb::transpose(q, {0, 2, 1, 3}); // [B, H, S, D]

		k = k.reshape(
				{k.shape()[0], k.shape()[1], 
				num_attention_heads_, head_dim_});
		k = mtb::transpose(k, {0, 2, 1, 3}); // [B, H, S, D]

		v = v.reshape(
				{v.shape()[0], v.shape()[1], 
				num_attention_heads_, head_dim_});
		v = mtb::transpose(v, {0, 2, 1, 3}); // [B, H, S, D]	

		// calculate attention scores
		// [B, H, S, D] * [B, H, D, S] -> [B, H, S, S]
		auto scores = mtb::matmul(q, mtb::transpose(k, {0, 1, 3, 2}));

		// apply mask
		auto scores_masked = apply_mask(scores);

		// softmax along the last dimension
		auto weights = mtb::softmax(scores_masked, -1); 

		// calculate attention output
		// [B, H, S, D]
		auto attn_output = mtb::matmul(weights, v); 
		
		// transpose back to  [B, S, H, D]
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

	void fill_k(T* data, int size) {
		k_proj_.fill_weight(data, size);
	}

	void fill_v(T* data, int size) {
		v_proj_.fill_weight(data, size);
	}

	void fill_q(T* data, int size) {
		q_proj_.fill_weight(data, size);
	}

	void fill_out(T* weights, T* bias, size_t w_size, size_t b_size) {
		out_proj_.fill_weight(weights, w_size);
		out_proj_.fill_bias(bias, b_size);
	}

protected:
	mtb::Tensor<T> apply_mask(const mtb::Tensor<T>& scores) {
		// apply the mask to the scores
		// scores [B, H, S, S] * mask [S, S] -> [B, H, S, S]
		auto mask_ = mtb::triu<uint8_t>(
				mtb::ones<uint8_t>(
                    {scores.shape()[2], scores.shape()[3]}), 1);
		
		auto masked_scores = scores.copy();
		for (size_t b = 0; b < scores.shape()[0]; ++b)
			for (size_t h = 0; h < scores.shape()[1]; ++h)
				for (size_t i = 0; i < scores.shape()[2]; ++i)
					for (size_t j = 0; j < scores.shape()[3]; ++j)
						if (mask_(i, j) == static_cast<uint8_t>(1))
								masked_scores(b, h, i, j) = T(-INFINITY);

		return masked_scores;
	}
	
private:
    int embed_size_;
    int num_attention_heads_;
    int head_dim_;

	MLinear<T> k_proj_;
	MLinear<T> v_proj_;
	MLinear<T> q_proj_;
	MLinear<T> out_proj_;
};

} // namespace mnn

#endif // __MSelfAT_HPP__
