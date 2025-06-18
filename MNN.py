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

class GPTNeoAttention(MModule):
    def __init__(self, embed_size, layer_id=0):
        self.layer_id = layer_id
        self.attention_type = ""
        
        self.attention = MSelfAT()

    def forward(self, hidden_states, k_h=None, v_h=None):
        # hidden_states shape: [B, S, E]
        return self.attention(hidden_states, k_h, v_h)
    
class GPTNeoMLP(MModule):
    def __init__(self, embed_dim, intermediate_size):
        embed_dim = embed_dim
        self.c_fc = MLinear(embed_dim, intermediate_size, bias=True)
        self.c_proj = MLinear(intermediate_size, embed_dim, bias=True)
        self.act = GELUActivation()
        
    def forward(self, hidden_states):
        # hidden_states shape: [B, S, E]
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        return hidden_states
    
class GPTNeoBlock(MModule):
    def __init__(self, embed_size, layer_id=0):
        self.layer_id = layer_id
        self.ln_1 = MLayerNorm(embed_size)
        self.attention = GPTNeoAttention(embed_size, layer_id)
        self.ln_2 = MLayerNorm(embed_size)
        self.mlp = GPTNeoMLP(embed_size, embed_size * 4)
        
    def forward(self, hidden_states, k_h=None, v_h=None):
        # hidden_states shape: [B, S, E]
        residual = hidden_states.copy()  # Save the input for residual connection
        hidden_states = self.ln_1(hidden_states)
        hidden_states, k_h_n, v_h_n = self.attention(hidden_states, k_h, v_h)  # Apply attention
        hidden_states += residual
        
        residual = hidden_states.copy()  # Save the input for residual connection
        hidden_states = self.ln_2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states += residual
        
        return hidden_states, k_h_n, v_h_n  # Return hidden states and key/value states for caching
    
class GPTNeoModel(MModule):
    def __init__(self, num_layers, embed_size = 768):
        self.num_layers = num_layers
        self.embed_size = embed_size
        vocab_size = 50257  # Common vocabulary size for GPT models
        self.wte = MEmbed(vocab_size, embed_size)
        self.wpe = MEmbed(2048, embed_size)  # Position embeddings, assuming max position embeddings of 2048
        
        self.layers = [GPTNeoBlock(embed_size, layer_id=i) for i in range(num_layers)]
        self.ln_f = MLayerNorm(embed_size)
        
        self.lm_head = MLinear(embed_size, vocab_size, bias=False)  # Language model head for output logits
        
    
    def forward(self, hidden_states, cache = None, position_ids=None):
        input_embeddings = self.wte(hidden_states) # Convert token IDs to embeddings
        
        if cache is None:
            sequence_length = input_embeddings.shape[1]
            position_ids = np.arange(sequence_length).reshape(1, -1)
        else:
            # position_ids 
            # position_ids = np.arange(sequence_length + 1).reshape(1, -1)
            pass
        # print("position_ids shape:", position_ids.shape)
        # print("input_embeddings shape:", input_embeddings.shape)
        
        
        position_embeddings = self.wpe(position_ids)  # Get position embeddings
        hidden_states = input_embeddings + position_embeddings
        
        for layer in self.layers:
            if cache is not None:
                k_h, v_h = cache.get(layer.layer_id, (None, None))  # Get cached key and value states
            else:
                k_h, v_h = None, None
            
            hidden_states, k_h_n, v_h_n = layer(hidden_states, k_h, v_h)  # Pass through each transformer block
            if cache is not None:
                cache[layer.layer_id] = (k_h_n, v_h_n)  # Update cache with new key and value states
        
        hidden_states = self.ln_f(hidden_states)
        # Get logits for the next token prediction
        output_logits = self.lm_head(hidden_states) 
        return output_logits # Shape: [B, S, vocab_size]

    def load(self, path="./model_state_dict"):
        # init wte 
        self.wte.weight = np.load(f"{path}/transformer/wte/weight/parameters.npy")
        print("model fp:",self.wte.weight.dtype)
        # init wpe
        self.wpe.weight = np.load(f"{path}/transformer/wpe/weight/parameters.npy")
        # init layers
        for i, layer in enumerate(self.layers):
            layer.attention.attention.k_proj.weight = np.load(f"{path}/transformer/h/{i}/attn/attention/k_proj/weight/parameters.npy")
            layer.attention.attention.v_proj.weight = np.load(f"{path}/transformer/h/{i}/attn/attention/v_proj/weight/parameters.npy")
            layer.attention.attention.q_proj.weight = np.load(f"{path}/transformer/h/{i}/attn/attention/q_proj/weight/parameters.npy")
            layer.attention.attention.out_proj.weight = np.load(f"{path}/transformer/h/{i}/attn/attention/out_proj/weight/parameters.npy")
            layer.attention.attention.out_proj.bias = np.load(f"{path}/transformer/h/{i}/attn/attention/out_proj/bias/parameters.npy")
            
            layer.mlp.c_fc.weight = np.load(f"{path}/transformer/h/{i}/mlp/c_fc/weight/parameters.npy")
            layer.mlp.c_fc.bias = np.load(f"{path}/transformer/h/{i}/mlp/c_fc/bias/parameters.npy")
            layer.mlp.c_proj.weight = np.load(f"{path}/transformer/h/{i}/mlp/c_proj/weight/parameters.npy")
            layer.mlp.c_proj.bias = np.load(f"{path}/transformer/h/{i}/mlp/c_proj/bias/parameters.npy")
            # init ln_1
            layer.ln_1.gamma = np.load(f"{path}/transformer/h/{i}/ln_1/weight/parameters.npy")
            layer.ln_1.beta = np.load(f"{path}/transformer/h/{i}/ln_1/bias/parameters.npy")       
            # init ln_2
            layer.ln_2.gamma = np.load(f"{path}/transformer/h/{i}/ln_2/weight/parameters.npy")
            layer.ln_2.beta = np.load(f"{path}/transformer/h/{i}/ln_2/bias/parameters.npy")
            
        # init ln_f
        self.ln_f.gamma = np.load(f"{path}/transformer/ln_f/weight/parameters.npy")
        self.ln_f.beta = np.load(f"{path}/transformer/ln_f/bias/parameters.npy")
        
        # init lm_head
        self.lm_head.weight = np.load(f"{path}/lm_head/weight/parameters.npy")
        