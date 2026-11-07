# Old hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 5000      # max_iters is used to train the optimizer, based on the loss be generated
eval_interval = 500   # how much loss is generated in above every iteration
learning_rate = 1e-3

# This help to use the GPU or else CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200        # point 4

n_embd = 32  # no. of embedding dimension
# ------------