import json

class MiniTokenizer:
    def __init__(self, vocab_file, merge_file):
        self.bpe_ranks = self.load_bpe_ranks(merge_file)
        self.vocab2id, self.id2vocab = self.load_vocab(vocab_file)
        self.bos_token = self.vocab2id.get('<|endoftext|>')
        self.eos_token = self.vocab2id.get('<|endoftext|>')
        self.unk_token = self.vocab2id.get('<|endoftext|>')
        
    def load_bpe_ranks(self, merge_file):
        bpe_ranks = {}
        with open(merge_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i == 0:
                    # the first line is the header
                    continue 
                parts = line.strip().split()
                if len(parts) != 2:
                    continue
                bpe_ranks[(parts[0], parts[1])] = i - 1
        return bpe_ranks
                
    def load_vocab(self, vocab_file):
        with open(vocab_file, 'r', encoding='utf-8') as f:
            vocab2id = json.load(f)
            
        id2vocab = {v: k for k, v in vocab2id.items()}
        return vocab2id, id2vocab
    
    def get_pairs(self, chars):
        """Get all adjacent character pairs in the list."""
        pairs = set()
        prev_char = chars[0]
        for char in chars[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs
    
    def bpe_tokenize(self, text):
        # split the text into characters
        chars = list(text)
        while True:
            pairs = self.get_pairs(chars)
            if not pairs:
                # no pairs left, all mereged
                break
            # find the most high-ranked pair, 
            # smallest index in bpe_ranks
            best_pair = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if best_pair not in self.bpe_ranks:
                break
            
            # merge the best pair
            new_chars = []
            i = 0
            while i < len(chars):
                if i < len(chars) - 1 and (chars[i], chars[i + 1]) == best_pair:
                    # merge the pair
                    new_chars.append(''.join(best_pair))
                    i += 2
                else:
                    # keep the character as is
                    new_chars.append(chars[i])
                    i += 1
            # move to the next iteration
            chars = new_chars
        return chars
    
    def tokenize(self, text):
        # fast handling of special tokens
        # replace the empty space with a special token 'Ġ'
        text = text.replace(' ', 'Ġ')
        return self.bpe_tokenize(text)  

    def encode(self, text):
        tokens = self.tokenize(text)
        # convert tokens to ids
        token_ids = [self.vocab2id.get(token, self.unk_token) for token in tokens]
        return token_ids

    def decode(self, token_ids):
        # convert ids to tokens
        tokens = [self.id2vocab.get(id, self.id2vocab.get(self.unk_token)) for id in token_ids]
        # join the tokens into a string
        text = ''.join(tokens)
        # replace the special token 'Ġ' with space
        text = text.replace('Ġ', ' ')
        text = text.replace('Ċ', '\n')  # handle new line if needed
        return text