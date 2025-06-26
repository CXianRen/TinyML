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
		  out_proj_(embed_size_, embed_size_),
		  mask_(mtb::triu<bool>(mtb::ones<bool>({2048, 2048}), 1)) 
		{}
		// Initialize the projection layers

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
		
		return scores;
	}

	
private:
    int embed_size_;
    int num_attention_heads_;
    int head_dim_;

	MLinear<T> k_proj_;
	MLinear<T> v_proj_;
	MLinear<T> q_proj_;
	MLinear<T> out_proj_;

	mtb::Tensor<bool> mask_;
};

} // namespace mnn

#endif // __MSelfAT_HPP__
