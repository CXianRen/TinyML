from miniTokenizer import MiniTokenizer

from transformers import AutoTokenizer

def compare_tokens(tokens, tokens_t):
    if len(tokens) != len(tokens_t):
        print("Token lengths differ:", len(tokens), "vs", len(tokens_t))
        return False
    for i in range(len(tokens)):
        if tokens[i] != tokens_t[i]:
            print(f"Token mismatch at index {i}: {tokens[i]} vs {tokens_t[i]}")
            return False
    print("Tokens match!")
    return True


def test():
    # 
    tokenizer_t = AutoTokenizer.from_pretrained("roneneldan/TinyStories-33M")

    # Load the tokenizer
    tokenizer = MiniTokenizer(
        vocab_file="vocab.json",
        merge_file="merges.txt")

    # Example sentence
    sentence = "There is a cat on the roof."

    # Tokenize the sentence
    tokens = tokenizer.tokenize(sentence)
    tokens_t = tokenizer_t.tokenize(sentence)

    # Print the tokens
    print("Tokens:", tokens)
    # compare the two tokenizers
    compare_tokens(tokens, tokens_t)
    

    # # Convert tokens to IDs
    token_ids = tokenizer.encode(sentence)
    token_ids_t = tokenizer_t.encode(sentence)
    # compare the two tokenizers
    compare_tokens(token_ids, token_ids_t)

    # decoded_sentence = tokenizer.decode(token_ids)
    decoded_sentence_t = tokenizer_t.decode(token_ids_t)
    decoded_sentence = tokenizer.decode(token_ids)
    if decoded_sentence != decoded_sentence_t:
        print("Decoded sentences do not match!")
        print("TinyTokenizer:", decoded_sentence)
        print("TransformersTokenizer:", decoded_sentence_t)
    else:
        print("Decoded sentences match!")
        print("Decoded sentence:", decoded_sentence)
    
test()