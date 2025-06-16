import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from myAT import GPTNeoModel
import numpy as np

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


import torch

# def greedy_generate(model, tokenizer, input_ids, max_new_tokens=100):
#     model.eval()
#     generated = input_ids
#     past_key_values = None

#     with torch.no_grad():
#         for idx in range(max_new_tokens):
#             # 只输入最新生成的 token，和缓存
#             outputs = model(input_ids=generated[:, -1:], past_key_values=past_key_values)
#             next_token_logits = outputs.logits[:, -1, :]
#             past_key_values = outputs.past_key_values

#             next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
#             generated = torch.cat((generated, next_token), dim=-1)

#             if next_token.item() == tokenizer.eos_token_id:
#                 break
#             # decode the new token and print it
#             print(tokenizer.decode(next_token.item(), skip_special_tokens=True), end='', flush=True)
#             # print(idx, end=' ', flush=True)
#     return generated

# def greedy_generate_no_cache(model, tokenizer, input_ids, max_new_tokens=100):
#     model.eval()
#     generated = input_ids

#     with torch.no_grad():
#         for idx in range(max_new_tokens):
#             # 每次输入整个序列，不使用 past_key_values
#             outputs = model(input_ids=generated)
#             next_token_logits = outputs.logits[:, -1, :]  # 取最后一个 token 的 logits

#             next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
#             generated = torch.cat((generated, next_token), dim=-1)

#             if next_token.item() == tokenizer.eos_token_id:
#                 break

#             print(tokenizer.decode(next_token.item(), skip_special_tokens=True), end='', flush=True)
#     print() 
#     return generated

myModel = GPTNeoModel(num_layers=4)
myModel.load()

def greedy_generate_no_cache(model, tokenizer, input_ids, max_new_tokens=100):
    # numpy version
    generated = input_ids
    
    with torch.no_grad():
        for idx in range(max_new_tokens):
            # 每次输入整个序列，不使用 past_key_values
            # input and output are numpy arrays
            outputs = model(generated)
            next_token_logits = outputs[:, -1, :]  # 取最后一个 token 的 logits

            # next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            # generated = torch.cat((generated, next_token), dim=-1)
            next_token = np.argmax(next_token_logits, axis=-1).reshape(-1, 1)
            generated = np.concatenate((generated, next_token), axis=-1)
            
            if next_token == tokenizer.eos_token_id:
                break

            print(tokenizer.decode(next_token.item(), skip_special_tokens=True), end='', flush=True)
    print() 
    return generated

input_ids = tokenizer("Once upon a time", return_tensors="np").input_ids  # use numpy
input_ids = input_ids.astype(np.int64)  # ensure the type is int64 for numpy
generated_ids = greedy_generate_no_cache(myModel, tokenizer, input_ids, max_new_tokens=20)

# input_ids = tokenizer("Once upon a time", return_tensors="pt").input_ids.to(model.device)
# generated_ids = greedy_generate_no_cache(model, tokenizer, input_ids, max_new_tokens=20)
# print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))