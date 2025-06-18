import numpy as np


class MModule():
    def __init__(self):
        self.forward = None
    
    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)

class MLinear(MModule):
    def __init__(self, in_features, out_features, bias=True):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = np.random.rand(out_features, in_features)
        
        if bias:
            self.bias = np.random.rand(out_features)
        else:
            self.bias = None

        self.forward = self.forward_with_bias if self.bias is not None else self.forward_without_bias

    def forward_with_bias(self, x):
        # x shape: [B, S, E]
        return np.dot(x, self.weight.T) + self.bias

    def forward_without_bias(self, x):
        # x shape: [B, S, E]
        return np.dot(x, self.weight.T)

class MSelfAT(MModule):
    def __init__(self):
        self.embed_size = 768
        self.num_attention_heads = 16
        self.head_dim = self.embed_size // self.num_attention_heads
        
        self.k_proj = MLinear(self.embed_size, self.embed_size, bias=False)
        self.v_proj = MLinear(self.embed_size, self.embed_size, bias=False)
        self.q_proj = MLinear(self.embed_size, self.embed_size, bias=False)
        self.out_proj = MLinear(self.embed_size, self.embed_size, bias=True)
        
        # this maks is pregenerated for the attention mask
        self.mask = np.triu(np.ones((2048, 2048)), k=1).astype(bool).reshape(1, 1, 2048, 2048)
        

    def get_kvq(self, hidden_states):     
        k = self.k_proj(hidden_states)  # [B, S, E] [E, E] | [B, S + 1, E] [E, E]
        v = self.v_proj(hidden_states)  # [B, S, E]
        q = self.q_proj(hidden_states)  # [B, S, E]
        
        return k, v, q
    
        
    def split_heads(self, x):
        # Split the last dimension into (num_attention_heads, head_dim)
        batch_size, seq_length, _ = x.shape
        x = x.reshape(batch_size, seq_length, self.num_attention_heads, self.head_dim)
        # Transpose to [B, HN, S, HD]    
        return np.transpose(x, (0, 2, 1, 3))
    
    def _attn(self, query_states, key_states, value_states):
        # Compute attention scores
        # (QK^T)V
        
        # [B, HN, S, HD] @ [B, HN, HD, S] -> [B, HN, S, S]
        # [B, HN, 1, HD] @ [B, HN, HD, S+1] -> [B, HN, 1, S+1]
        # Q                K                   
        attention_scores = np.matmul(query_states, key_states.transpose(0, 1, 3, 2))

        # masked attention 
        query_len = query_states.shape[2]
        key_len = key_states.shape[2]
        mask = self.mask[:,:, key_len - query_len: key_len, :key_len]
        attention_scores = np.where(mask, -np.inf, attention_scores)
        
        # Apply softmax to get attention weights
        attention_weights = np.exp(attention_scores - np.max(attention_scores, axis=-1, keepdims=True))
        attention_weights /= np.sum(attention_weights, axis=-1, keepdims=True)
        
        # attention output  socre * value
        # [B, HN, S, S] @ [B, HN, S, HD] -> [B, HN, S, HD]
        # [B, HN, 1, S+1] @ [B, HN, S+1, HD] -> [B, HN, 1, HD]
        attention_output = np.matmul(attention_weights, value_states)
        
        return attention_output

    
    def forward(self, hidden_states, k_h = None, v_h = None):
        # [B, S, E]
        batch_size, seq_length, _ = hidden_states.shape 
        # Project inputs to key, value, query

        # [B, S, E][E, E] -> [B, S, E] 
        # next token:
        # [B, S + 1, E][E, E] -> [B, S + 1, E]
        k, v, q = self.get_kvq(hidden_states)  
        
        # Split heads
        # [B, S, E] -> [B, S, HN, HD] -> [B, HN, S, HD]
        # [B, S + 1, E] -> [B, HN, S + 1, HD]
        query_states = self.split_heads(q)  # [B, HN, S, HD]
        key_states = self.split_heads(k)    # [B, HN, S, HD]
        value_states = self.split_heads(v)  # [B, HN, S, HD]
        
        if k_h is not None and v_h is not None:
            # If k_h and v_h are provided, use them
            # [B, HN, 1, HD] -> [B, HN, S + 1, HD]

            # print("k_h shape:", k_h.shape, "v_h shape:", v_h.shape)
            # print("key_states shape:", key_states.shape, "value_states shape:", value_states.shape)
            key_states = np.concatenate((k_h, key_states), axis=2)  # [B, HN, S + 1, HD]
            value_states = np.concatenate((v_h, value_states), axis=2)
        
        # (QK^T)V 
        attention_output = self._attn(query_states, key_states, value_states)  # [B, HN, S, HD]
        
        # Combine heads back to the original shape
        # [B, HN, S, HD] -> [B, S, HN, HD]
        # [B, HN, 1, HD] -> [B, 1, HN, HD]
        attention_output = np.transpose(attention_output, (0, 2, 1, 3)).copy()
        # Reshape to [B, S, E]
        # [B, 1, HN, HD] -> [B, 1, E]
        attention_output = attention_output.reshape(batch_size, seq_length, self.embed_size)
        
        # out = np.dot(attention_output, self.out_proj.T) + self.out_bias
        out = self.out_proj(attention_output)
        # [B, S, E][E, E] -> [B, S, E]
        # [B, 1, E][E, E] -> [B, 1, E]
        
        return out, key_states, value_states  # Return attention output and key/value states for caching

class MLayerNorm(MModule):
    def __init__(self, normalized_shape, eps=1e-5):
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.gamma = np.ones(normalized_shape)
        self.beta = np.zeros(normalized_shape)

    def forward(self, x):
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.var(x, axis=-1, keepdims=True)
        normalized_x = (x - mean) / np.sqrt(variance + self.eps)
        return self.gamma * normalized_x + self.beta
    
class MEmbed(MModule):
    def __init__(self, vocab_size, embed_dim):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.weight = np.random.rand(vocab_size, embed_dim)

    def forward(self, input_ids):
        # input_ids shape: [B, S]
        return self.weight[input_ids]  # Select embeddings for the given input IDs

class GELUActivation(MModule):
    def __init__(self):
        pass
    
    def forward(self, x):
        # GELU(x) = 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715*x^3) ))
        sqrt_2_over_pi = np.sqrt(2.0 / np.pi)
        return 0.5 * x * (1.0 + np.tanh(sqrt_2_over_pi * (x + 0.044715 * np.power(x, 3.0))))        

