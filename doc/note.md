

# attention

token = 768
sentence = [N, 768]
<!-- [N + 1, 768] -->
batch = [B,  N , 768] 
<!-- [B,  N + 1 , 768]  -->
k,q,v = [768, 768]

<!-- project -->
[B, N, 768] [768, 768] 
<!-- [B, N + 1, 768] [768, 768]  -->
<!-- [B,     1, 768] [768, 768] -->


new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
[B, N, 768][:-1] = [B,N, NH(16), HS(768/16)]

<!-- [B, N + 1, 768]  -->


tensor = tensor.view(new_shape)
return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

[B, NH, N, HS]
<!-- [B, NH, N + 1, HS]  -->

attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

s = Softmax(QK)
s = [B, NH, N, HS] [B, NH, HS, N] = [B, NH, N, N]
<!--[B, NH, N + 1, HS] [B, NH, HS, N + 1] -> [B, NH, N + 1, N + 1]-->

out= sV = [B, NH, N, N] [B, NH, N, HS] =  [B, NH, N, HS]
<!-- [B, NH, N + 1, N + 1]  [B, NH, N + 1, HS] -->


attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)

[B, N, NH, HS]

[B, N, 768]

attn_output = self.out_proj(attn_output)

[B, N, 768]

attn_output = self.resid_dropout(attn_output)


#  Byte-Pair Encoding（BPE）

given a sentence: "There is a cat."

## First split it by word then split it by char
```
There -> ['T', 'h', 'e', 'r', 'e', ...]
```



# op support

reshape (member function)
copy
contiguous

+
-
*
/

+=
-+
*=
/=


np.transpose (T or class function)

np.matmul
np.concatenate (class function)

np.where

np.ones
np.zeros
np.random.rand
np.triu


np.exp
np.max
np.min
np.sum
np.mean
np.var
np.sqrt
np.tanh

#
[S, HN, HD]
[HN, S, HD]

[1, 2, 3]

[
 [1, 2, 3]
 [4, 5, 6]
]

[2, 1, 3]
[
  [
    [1, 2, 3]
  ]
  [
    [4, 5, 6]
  ]
]


# broadcasting
[](https://numpy.org/doc/stable/user/basics.broadcasting.html)

1. scalar broadcast
2. 




b = a[[1,2,7]] // copy
b = a[1:3, 1:3] // share
b = b = a[1:2:6, 1:2:3] // share


# concat 
1. should have same dimensions
2. they should be same at all dimension except the concat one.
3. same type.

