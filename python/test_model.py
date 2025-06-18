from transformers import AutoTokenizer
from GPTNeo import GPTNeoModel
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("roneneldan/TinyStories-33M")
print("Tokenizer loaded.")
# the type of tokenizer
print("Tokenizer type:", type(tokenizer))


myModel = GPTNeoModel(num_layers=4)
myModel.load("../model_state_dict")  # load the model state dict

def greedy_generate_no_cache(model, tokenizer, input_ids, max_new_tokens=100):
    # numpy version
    generated = input_ids
    cache = None
    for idx in range(max_new_tokens):
        # input and output are numpy arrays
        if cache is None:
            outputs = model(generated, None)
        else:
            if idx == 0:
                outputs = model(generated, cache, 
                    position_ids=np.arange(
                        generated.shape[1]).reshape(1, -1))
            else:
                # print("input shape:", generated[:, -1:].shape)
                outputs = model(generated[:, -1:], cache, 
                    position_ids=np.arange(
                        generated.shape[1], generated.shape[1]+1).reshape(1, -1))
                
        next_token_logits = outputs[:, -1, :]  # get the last token logits

        next_token = np.argmax(next_token_logits, axis=-1).reshape(-1, 1)
        generated = np.concatenate((generated, next_token), axis=-1)
        
        if next_token == tokenizer.eos_token_id:
            break
        # print("output:", tokenizer.decode(next_token.item(), skip_special_tokens=True))
        # print("cache size:", cache[0][0].shape)
        print(tokenizer.decode(next_token.item(), skip_special_tokens=True), end='', flush=True)
    print() 
    return generated

input_ids = tokenizer("Once upon a time,", return_tensors="np").input_ids  # use numpy
input_ids = input_ids.astype(np.int64)  # ensure the type is int64 for numpy

import time
start_time = time.time()
generated_ids = greedy_generate_no_cache(myModel, tokenizer, input_ids, max_new_tokens=200)
end_time = time.time()
print(f"Average time per token: {(end_time - start_time) / 20:.4f} seconds")
# print(generated_ids)
