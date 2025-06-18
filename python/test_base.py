from MNN import MSelfAT, MLayerNorm, GELUActivation, MEmbed
import numpy as np
import torch
from transformers.activations import NewGELUActivation
from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoSelfAttention
from types import SimpleNamespace


def check_output(output, output_data, name):
    if np.allclose(output, output_data, atol=1e-5):
        print("{} output matches expected output.".format(name))
    else:
        print("{} output does not match expected output.".format(name))
        
           
def test_MSelfAT():
    attn = MSelfAT()
    attn.k_proj.weight = np.load('model_state_dict/transformer/h/1/attn/attention/k_proj/weight/parameters.npy')
    attn.v_proj.weight = np.load('model_state_dict/transformer/h/1/attn/attention/v_proj/weight/parameters.npy')
    attn.q_proj.weight = np.load('model_state_dict/transformer/h/1/attn/attention/q_proj/weight/parameters.npy')
    attn.out_proj.weight = np.load('model_state_dict/transformer/h/1/attn/attention/out_proj/weight/parameters.npy')
    attn.out_proj.bias = np.load('model_state_dict/transformer/h/1/attn/attention/out_proj/bias/parameters.npy')

    input_data = np.random.randn(2, 10, 768).astype(np.float32)  # [B, S, E]
    input_tensor = torch.tensor(input_data, dtype=torch.float32)
    
    config = SimpleNamespace(
        max_position_embeddings=2048,
        hidden_size=768,
        num_heads=16,
        attention_dropout = 0.0,
        resid_dropout = 0.0,
    )
    
    attn_t = GPTNeoSelfAttention(config=config, attention_type="global")
    attn_t.k_proj.weight = torch.nn.Parameter(torch.tensor(attn.k_proj.weight, dtype=torch.float32))
    attn_t.v_proj.weight = torch.nn.Parameter(torch.tensor(attn.v_proj.weight, dtype=torch.float32))
    attn_t.q_proj.weight = torch.nn.Parameter(torch.tensor(attn.q_proj.weight, dtype=torch.float32))
    attn_t.out_proj.weight = torch.nn.Parameter(torch.tensor(attn.out_proj.weight, dtype=torch.float32))
    attn_t.out_proj.bias = torch.nn.Parameter(torch.tensor(attn.out_proj.bias, dtype=torch.float32))
    
    with torch.no_grad():
        output_t, _ = attn_t(input_tensor)
    output_data = output_t.numpy()

    out, _, _ = attn.forward(input_data)  # [B, S, E]
    check_output(out, output_data, "Final Output")

def testLayerNorm():
    ln = MLayerNorm(normalized_shape=768)
    ln.gamma = np.load('model_state_dict/transformer/h/0/ln_1/weight/parameters.npy')
    ln.beta = np.load('model_state_dict/transformer/h/0/ln_1/bias/parameters.npy')
    
    input_tensor = torch.randn(1, 10, 768, dtype=torch.float32)
    
    ln_t = torch.nn.LayerNorm(normalized_shape=768, 
                              elementwise_affine=True,
                              eps=1e-5)
    ln_t.weight = torch.nn.Parameter(torch.tensor(ln.gamma, dtype=torch.float32))
    ln_t.bias = torch.nn.Parameter(torch.tensor(ln.beta, dtype=torch.float32))
    
    with torch.no_grad():
        output_t = ln_t(input_tensor)

    output = ln(input_tensor.numpy())
    check_output(output, output_t.numpy(), "LayerNorm Output")
    
def testGELUActivation():
    gelu = GELUActivation()
    
    input_tensor = torch.randn(1, 10, 768, dtype=torch.float32)
    
    gelu_t = NewGELUActivation()
    
    with torch.no_grad():
        output_t = gelu_t(input_tensor)

    output = gelu.forward(input_tensor.numpy())
    check_output(output, output_t.numpy(), "GELU Activation Output")

def testEmbedding():
    embed = MEmbed(vocab_size=50257, embed_dim=768)
    embed.weight = np.load('model_state_dict/transformer/wte/weight/parameters.npy')
    
    input_ids = np.array([[101, 102, 103], [104, 105, 106]])  # Example input IDs
    
    embed_t = torch.nn.Embedding(50257, 768)
    embed_t.weight = torch.nn.Parameter(torch.tensor(embed.weight, dtype=torch.float32))
  
    with torch.no_grad():
        output_t = embed_t(torch.tensor(input_ids, dtype=torch.int64))
    output_data = output_t.numpy()
    
    out = embed.forward(input_ids)  # [B, S, E]
    check_output(out, output_data, "Embedding Output")

def test_MSelfAT_with_hsitory():
    attn = MSelfAT()
    input_data = np.random.randn(1, 200, 768).astype(np.float32)  # [B, S, E]
    output_data, _, _ = attn.forward(input_data)  # [B, S, E]
    
    
    input_data_0_9 = input_data[:, :-2, :]
    _, k_h, v_h = attn.forward(input_data_0_9)  # [B, S, E]
    input_data_9_10 = input_data[:, -1:, :]
    output_data_9_10, k_h, v_h = attn.forward(input_data_9_10, k_h, v_h)  # [B, S, E]
    print("output_data shape:", output_data.shape)
    print("output_data_9_10 shape:", output_data_9_10.shape)
    check_output(output_data[:, -1:, :], output_data_9_10, "MSelfAT with history")
    
test_MSelfAT()
testLayerNorm()
testGELUActivation()
testEmbedding()
test_MSelfAT_with_hsitory()