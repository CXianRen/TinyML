from MNN import *

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
        