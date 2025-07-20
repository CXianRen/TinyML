from tokenizer.miniTokenizer import MiniTokenizer
from GPTNeo import GPTNeoModel
import numpy as np

tokenizer = MiniTokenizer(
    vocab_file="tokenizer/vocab.json",
    merge_file="tokenizer/merges.txt"
)
print("Tokenizer loaded.")


myModel = GPTNeoModel(num_layers=4)
myModel.load("../model_state_dict")  # load the model state dict

print("Model loaded.")

def greedy_generate(model, tokenizer, input_ids, max_new_tokens=100):
    # numpy version
    generated = input_ids
    cache = {}
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
        
        if next_token == tokenizer.eos_token:
            break
        # print("output:", tokenizer.decode(next_token.item(), skip_special_tokens=True))
        # print("cache size:", cache[0][0].shape)
        print(tokenizer.decode([next_token.item()]), end='', flush=True)
    print() 
    return generated

input_ids = tokenizer.encode("Once upon a time,")
input_ids = np.array(input_ids).reshape(1, -1)  # reshape to 2D array
input_ids = input_ids.astype(np.int64)  # ensure the type is int64 for numpy

import time
start_time = time.time()
generated_ids = greedy_generate(myModel, tokenizer, input_ids, max_new_tokens=100)
end_time = time.time()
print(f"Average token per second: {100 / (end_time - start_time):.2f}")
# print(generated_ids)
