from transformers import AutoTokenizer
from myAT import GPTNeoModel
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("roneneldan/TinyStories-33M")

myModel = GPTNeoModel(num_layers=4)
myModel.load()

def greedy_generate_no_cache(model, tokenizer, input_ids, max_new_tokens=100):
    # numpy version
    generated = input_ids
    
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
generated_ids = greedy_generate_no_cache(myModel, tokenizer, input_ids, max_new_tokens=200)
