# this is just for testing gpt code for different input data

import torch
import torch.nn as nn
from torch.nn import functional as F

# Scaled-up hyperparameters :
batch_size = 64 # how many independent sequences will we process in parallel?  (32 --> 64)
block_size = 256 # what is the maximum context length for predictions? (8 --> 256)
max_iters = 5000      # max_iters is used to train the optimizer, based on the loss be generated
eval_interval = 500   # how much loss is generated in above every iteration
learning_rate = 3e-4  # standard transformer learning rate

# This help to use the GPU or else CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"device: {device}")
eval_iters = 200        # point 4

n_embd = 384    # no. of embedding dimension, 384 features per token (32 --> 384)
n_head = 6      # 6 attention heads (384 // 6 = 64 head_size)
n_layer = 6     # 6 transformer blocks deep (was 3)
dropout = 0.2   # 20% dropout
# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('testingrocketdata.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    #  V IMP : generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)   # this load the data in the GPU, as device is used
    return x, y

@torch.no_grad()                   # point 5
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
        # One single head  of self-attention

    def __init__(self, head_size):
        super().__init__()

        # Remember n_embd = C (channels) = 32, which we get after tok_emb-->+pos_emb = give (4,8,32)-->lm_head gives logits of shape(4, 8, 65)
        self.key = nn.Linear(n_embd, head_size, bias=False)    # (4, 8, 32)  change made  32 --> 16
        self.query = nn.Linear(n_embd, head_size, bias=False)  # (4, 8, 32)  change made  32 --> 16
        self.value = nn.Linear(n_embd, head_size, bias=False)  # (4, 8, 32)  change made  32 --> 16
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)   # randomly prevent some tokens from communicating, temporary shutting off some neurons in neural network

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)    # (B,T,C=16)   # the C size is changed from 32 to 16
        q = self.query(x)  # (B,T,C=16)

        # compute attention scores ("affinities")
        # original --> wei = q @ k.transpose(-2, -1) * C**-0.5   # (B, T, T)
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5) # (B=4, T=8, H=C=16) @ (B=4, H=C=16, T=8) ---> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))   # (B, T, T)
        wei = F.softmax(wei, dim=-1)   # (B, T, T)

        wei = self.dropout(wei)

        # perform the weighted aggregation of the values
        v = self.value(x)   # (B, T, C=16)
        out = wei @ v       # (B, T, T) @ (B, T, C) --> (B, T, C=16)

        return out

# Multi-head attention make more 4 heads, so that those heads learned different things for the input given, like, grammar, verbs, etc,
class MultiHeadAttention(nn.Module):
    # Multiple heads of self-attention in parallel

    def __init__(self, num_heads, head_size):
        super().__init__()
        # Create a list of independent Head instances
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)  # proj helps heads from the multi-head attention to talk to each other
        # as head_size (8) = n_embd (32) / num_head (4)

        # (num_heads * head_size) is replaced by n_embd,
        # It projects the concatenated heads back into the
        # residual pathway so the dimensions and features blend smoothly back into x.

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Run every head on x in parallel, and concatenate results on the channel dimension
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(self.proj(out))
        return out

# FeedForward lets the output got / produced by the heads, to talk to other heads. so, that they can get other head's learning pattern.
class FeedForward(nn.Module):
    # A simple linear layer followed by non-linearity

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),  # Expand: 32 --> 128  i.e. output of each  multi-head combine with feedforward is passed through block
            nn.ReLU(),                      # Non-linearity
            nn.Linear(4 * n_embd, n_embd),   # Project back: 128 --> 32
            nn.Dropout(dropout)  # shutting some neurons in teh neural network, so that when switch-on, again the more actively perform themself
        )

    def forward(self, x):
        return self.net(x)

# block will pass the output got by combining multi-head attention added with x(input) and followed(pass) by feedforward added with x(input),
# so, that the weights does not loss its efficiency during backpropogation (i.e. training)
class Block(nn.Module):
    # Transformer block : communication followed by computation

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head   # now, instead of passing the 32 head_size(Channels), we will pass the number of heads for each

        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # Notice the residual additions (x + ...) and LayerNorms:
        #x = x + self.sa(self.ln1(x))  # Communicate with residual highway
        #x = x + self.ffwd(self.ln2(x))  # Think with residual highway

        #x = x + self.sa(x)
        #x = x + self.ffwd(x)

        x = x + self.sa(self.ln1(x))  # applying layer-norm before attention and then doing attention
        x = x + self.ffwd(self.ln2(x))  # applying layer-norm before feed-forward and then doing feed-forward
        return x

# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table

        # Embedding table: purely for REPRESENTING tokens richly
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)  # Shape: (65, 32) — 65 tokens, each with 32 rich features

        # Position embedding — WHERE each token is
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        # 1. Before it was (single head):
        # self.sa_head = Head(n_embd)    # n_embd = head_size = 32
        # 2. Now (multi-head): 4 heads of size 8 = 32
        self.sa_heads = MultiHeadAttention(4, n_embd // 4)  # i.e. 4 heads of 8-dimensional self attention
        # 3. FeedForward: tokens think individually
        #self.ffwd = FeedForward(n_embd)

        # the separately made block has been removed instead this is added
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # Final layer norm

        # NEW: A separate Linear layer purely for PREDICTING next token
        self.lm_head = nn.Linear(n_embd, vocab_size)  # Takes 32 features → outputs 65 logits (one per character)

    def forward(self, idx, targets=None):

        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers

        # Step 1: Lookup rich 32-dimensional embedding for each token
        tok_emb = self.token_embedding_table(idx)  # (B, T, n_embd) = (B, T, 32), tok_emb = token will be represented from the embedding table

        # Step 2 : tell's the position of the input token, with the help of the above look-up table
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T, C)
        # torch.arange(T) = [0, 1, 2, 3, 4, 5, 6, 7] ← position indices

        # ADD them together! Each token now knows WHAT it is AND WHERE it is
        x = tok_emb + pos_emb  # (B, T, C)

        # 1. self attention is changed to multi-head attention
        #x = self.sa_head(x)  # apply one head of self-attention (B, T, C)

        # 2. Applying multi-head attention
        #x = self.sa_heads(x)  # (B, T, 32)  <-- all 4 heads communicate and combine!
        # it is changed to blocks directly

        # 3. Tokens think individually
        #x = self.ffwd(x)  # (B, T, C)
        # they thinking has now been combined!

        x = self.blocks(x)

        # Step 3: Project 32 features → 65 vocabulary logits
        logits = self.lm_head(x)  # (B, T, vocab_size) = (B, T, 65), logits are generated for each character

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop index to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel()
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets       # point 7
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


torch.save(model.state_dict(), 'rocketdatatest.pt')   # saving the weights and biases
#print("Model saved successfully as gpt_model.pt!")

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))