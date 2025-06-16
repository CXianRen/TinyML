import os
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("roneneldan/TinyStories-33M")
model = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-33M")

# print the model structure
print(model, file=open("model_structure.txt", "w"))


# 打印 Tokenizer 类型
print(type(tokenizer))  

# 查看特殊 Token
print("Special tokens:")
print("PAD:", tokenizer.pad_token)
print("BOS:", tokenizer.bos_token)
print("EOS:", tokenizer.eos_token)
print("UNK:", tokenizer.unk_token)

# 查看 vocab size
print("Vocab size:", tokenizer.vocab_size)

# 查看 tokenizer 的配置
print("Tokenizer config:", tokenizer.init_kwargs)

print(model.config, file=open("model_config.txt", "w"))

# model = model.transformer.h[0].ln_1
model = model.to("cpu")  # move to CPU for saving state_dict

import numpy as np
state_dict = model.state_dict()
for k, v in state_dict.items():
    print(f"{k}: {v.shape}")
    paht=k.replace(".", "/")
    os.makedirs(f"model_state_dict/{paht}", exist_ok=True)
    # save the state_dict to a file
    np.save(f"model_state_dict/{paht}/parameters.npy", v.cpu().numpy())
    
    # save the state_dict to numPy format
    # np.save(f"model_state_dict/{k}.npy", v.cpu().numpy())

# generate a sample input 
# import torch
# input_tensor = torch.randn(1, 10, 768, dtype=torch.float32)

# os.makedirs("model_state_dict/test", exist_ok=True)
# # save the input tensor to a file
# np.save("model_state_dict/test/input_tensor.npy", input_tensor.cpu().numpy())

# forward pass through the attention layer
# with torch.no_grad():
#     output = model(input_tensor)[0]
#     q = model.attention.q_proj(input_tensor)
#     k = model.attention.k_proj(input_tensor)
#     v = model.attention.v_proj(input_tensor)
    
# # save the output tensor to a file
# np.save("model_state_dict/test/output_tensor.npy", output.cpu().numpy())
# np.save("model_state_dict/test/q_tensor.npy", q.cpu().numpy())
# np.save("model_state_dict/test/k_tensor.npy", k.cpu().numpy())
# np.save("model_state_dict/test/v_tensor.npy", v.cpu().numpy())

# # attention_weights
# qs=model.attention._split_heads(q, 16, 768//16)
# ks=model.attention._split_heads(k, 16, 768//16)
# vs=model.attention._split_heads(v, 16, 768//16)
# attn_weights = torch.matmul(qs, ks.transpose(-1, -2))
# # save the attention weights to a file
# np.save("model_state_dict/test/attn_weights.npy", attn_weights.cpu().numpy())

# attn_output = model.attention._attn(qs, ks, vs)[0]
# # save the attention output to a file
# np.save("model_state_dict/test/attn_output.npy", attn_output.cpu().numpy())

# prompt = "Once upon a time"
# input_ids = tokenizer(prompt, return_tensors="pt").input_ids

# output = model.generate(input_ids, max_new_tokens=100)
# print(tokenizer.decode(output[0], skip_special_tokens=True))

# import torch

# def greedy_generate(model, tokenizer, input_ids, max_new_tokens=100):
#     model.eval()
#     generated = input_ids
#     past_key_values = None

#     with torch.no_grad():
#         for _ in range(max_new_tokens):
#             # 只输入最新生成的 token，和缓存
#             outputs = model(input_ids=generated[:, -1:], past_key_values=past_key_values)
#             next_token_logits = outputs.logits[:, -1, :]
#             past_key_values = outputs.past_key_values

#             next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
#             generated = torch.cat((generated, next_token), dim=-1)

#             if next_token.item() == tokenizer.eos_token_id:
#                 break
#     return generated

# input_ids = tokenizer("Once upon a time", return_tensors="pt").input_ids.to(model.device)
# generated_ids = greedy_generate(model, tokenizer, input_ids, max_new_tokens=50)
# print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))