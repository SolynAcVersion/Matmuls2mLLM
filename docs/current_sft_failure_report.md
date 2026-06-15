# Current SFT Failure Report

Date: 2026-06-12

This note records only the latest observed SFT/HiRA behavior and the current
code evidence. It intentionally excludes earlier fixed tokenizer/special-token
issues.

## Current Setup Under Discussion

Model family:

```text
GPT2-Medium style LM
24 layers
1024 hidden
16 heads
FFN 2752
RoPE
```

Current SFT path in `sft_chat_templetes.py`:

```text
pretrained/chat-vocab model
-> replace attention q/k/v/o projections with HiRALinear
-> train HiRA A/B parameters plus added special-token rows
-> resume from existing SFT checkpoint
```

## Model And Runtime Details

Reported training history for the base model:

```text
OWT pretraining
-> Wiki continue-pretraining
-> about 4B tokens total
-> SFT
-> HiRA continuation
```

Default pretrain checkpoint path in the current script:

```python
PRETRAIN_CKPT_PATH = os.environ.get(
    "PRETRAIN_CKPT_PATH",
    './data/pretrain_gpt2med_iter_390000.pt',
)
```

Code location:

```text
sft_chat_templetes.py:22-25
```

The script asserts checkpoint vocab size instead of resizing model weights inside
SFT:

```python
embed_vocab = state_dict["token_embeddings.embedding_weights"].shape[0]
lm_vocab = state_dict["lm_head.W"].shape[1]
...
if ckpt_vocab_size != tokenizer_vocab_size:
    raise ValueError(...)
```

Code location:

```text
sft_chat_templetes.py:106-123
```

### Tokenizer And Vocab

Current tokenizer construction:

```python
BASE_VOCAB_SIZE = 32000
BASE_VOCAB_PATH = './data/owt_train_32004.pickle'
MERGES_PATH = './data/owt_train_32000_merges.pickle'
CHAT_SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|user|>",
    "<|assistant|>",
    "<|pad|>",
]
```

Code location:

```text
sft_chat_templetes.py:19-31
```

The SFT vocab is built by copying base ids `0..31999`, then adding missing chat
special tokens:

```python
sft_vocab = {i: vocab[i] for i in range(base_vocab_size)}
...
next_id = base_vocab_size
for token in special_tokens:
    if token_bytes in token_to_id:
        continue
    sft_vocab[next_id] = token_bytes
    next_id += 1
```

Code location:

```text
sft_chat_templetes.py:55-89
```

Observed/expected special token ids for the current chat vocab:

```text
31999 <|endoftext|>
32000 <|user|>
32001 <|assistant|>
32002 <|pad|>
```

Therefore:

```text
base vocab size = 32000
chat SFT vocab size = 32003
```

### Transformer Architecture

Current SFT model config:

```python
context_length = 1024
batch_size = 8
d_model = 1024
num_layers = 24
num_heads = 16
d_ff = 2752
rope_theta = 10000
```

Code location:

```text
sft_chat_templetes.py:299-306
```

Derived architecture details:

```text
d_head = d_model / num_heads = 64
RMSNorm eps = 1e-5
position encoding = RoPE on q/k
attention = causal scaled_dot_product_attention
block style = pre-norm residual attention + pre-norm residual FFN
FFN = SwiGLU
lm_head = separate untied Linear, not tied to token embedding
```

Attention implementation:

```python
self.q_proj = nn.Linear(d_model, d_model)
self.k_proj = nn.Linear(d_model, d_model)
self.v_proj = nn.Linear(d_model, d_model)
self.o_proj = nn.Linear(d_model, d_model)
...
q = self.rope(q, token_positions)
k = self.rope(k, token_positions)
out = F.scaled_dot_product_attention(..., is_causal=True)
```

Code location:

```text
modules.py:308-379
```

Transformer block implementation:

```python
h = x + self.attn(self.ln1(x), token_positions)
out = h + self.ffn(self.ln2(h))
```

Code location:

```text
modules.py:950-1021
```

LM implementation:

```python
self.token_embeddings = Embedding(vocab_size, d_model)
self.layers = nn.ModuleList([... for _ in range(num_layers)])
self.ln_final = RMSNorm(d_model)
self.lm_head = Linear(d_model, vocab_size)
...
x = self.token_embeddings(in_indices)
for layer in self.layers:
    x = layer(x, token_positions=token_positions)
x = self.ln_final(x)
logits = self.lm_head(x)
```

Code location:

```text
modules.py:1023-1107
```

Implementation details that matter for parameter counting:

```text
Attention q/k/v/o use torch.nn.Linear, default bias=True.
The custom FFN Linear has no bias.
The custom lm_head stores W with shape (d_model, vocab_size).
The token embedding and lm_head are separate matrices.
```

Runtime numeric settings:

```python
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
...
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    logits = model(x)
```

Code locations:

```text
sft_chat_templetes.py:16-17
sft_chat_templetes.py:505-511
```

### SFT Hyperparameters

Current local HiRA SFT hyperparameters:

```python
max_iters = 41000
eval_interval = 500
eval_iters = 50
log_interval = 50
checkpoint_interval = 5000

max_learning_rate = 1e-5
min_learning_rate = 2e-6
warmup_iters = 2000
cosine_cycle_iters = max_iters

weight_decay = 0.0
betas = (0.9, 0.95)
eps = 1e-8
max_grad_norm = 0.1
```

Code location:

```text
sft_chat_templetes.py:308-323
```

Tokens per training iteration:

```text
batch_size * context_length * world_size = 8 * 1024 * 1 = 8192 tokens/iter
```

At `iter 35000`, printed processed tokens are:

```text
35000 * 8192 = 286,720,000 tokens ~= 0.287B
```

This matches the user's latest log.

Data split logic:

```python
valid_end = math.floor(len(json_data) * 0.1)
if for_valid:
    data = json_data[: valid_end]
else:
    data = json_data[valid_end: ]
```

Code location:

```text
sft_chat_templetes.py:221-226
```

Loss details:

```text
ignore_index = -666
loss = causal LM next-token cross entropy
prompt tokens are masked
assistant output tokens and final <|endoftext|> are supervised
```

Code locations:

```text
sft_chat_templetes.py:196-245
modules.py:715-728
```

### HiRA Configuration

Current local HiRA replacement:

```python
for l in model.layers:
    l.attn.q_proj = HiRALinear(l.attn.q_proj)
    l.attn.k_proj = HiRALinear(l.attn.k_proj)
    l.attn.v_proj = HiRALinear(l.attn.v_proj)
    l.attn.o_proj = HiRALinear(l.attn.o_proj)
```

Code location:

```text
sft_chat_templetes.py:376-380
```

Only attention projections are adapted:

```text
adapted: q_proj, k_proj, v_proj, o_proj in all 24 layers
not adapted: FFN w1/w2/w3, RMSNorms, base embeddings, base lm_head columns
```

Current local `HiRALinear`:

```python
class HiRALinear(nn.Module):
    def __init__(self, base, r=512, alpha=8):
        self.W_0 = nn.Parameter(weight.detach().clone(), requires_grad=False)
        self.bias = nn.Parameter(bias.detach().clone(), requires_grad=False)
        self.scale = alpha / r
        self.A = nn.Parameter(torch.randn(self.d_in, r) * 0.001)
        self.B = nn.Parameter(torch.zeros(r, self.d_out))
```

Code location:

```text
sft_chat_templetes.py:263-288
```

Trainability logic:

```python
for param in model.parameters():
    param.requires_grad = False

for n, p in model.named_parameters():
    if ".A" in n or ".B" in n:
        p.requires_grad = True

enable_added_token_training(model)
```

Code location:

```text
sft_chat_templetes.py:386-393
```

Added-token training:

```python
model.token_embeddings.embedding_weights.requires_grad = True
model.lm_head.W.requires_grad = True

grad[:base_vocab_size] = 0
grad[:, :base_vocab_size] = 0
```

Code location:

```text
sft_chat_templetes.py:125-140
```

Important implication:

```text
PyTorch reports the full embedding matrix and full lm_head matrix as trainable,
but gradient hooks zero out the base 32000 rows/columns. In effective terms,
only the three added special-token rows/columns are allowed to update.
```

### Parameter Counts

Assumptions for the counts below:

```text
vocab_size = 32003
d_model = 1024
num_layers = 24
num_heads = 16
d_ff = 2752
attention q/k/v/o have bias
embedding and lm_head are untied
```

Base chat-vocab model without HiRA:

```text
token_embeddings: 32003 * 1024 = 32,771,072
lm_head: 1024 * 32003 = 32,771,072
one transformer layer:
  attention q/k/v/o weights+biases = 4 * (1024*1024 + 1024) = 4,198,400
  SwiGLU FFN w1/w2/w3 = 3 * (1024*2752) = 8,454,144
  RMSNorm ln1+ln2 = 2 * 1024 = 2,048
  layer total = 12,654,592
24 layers = 303,710,208
ln_final = 1,024

base total = 369,253,376 parameters
```

HiRA `r=8, alpha=8` case:

```text
HiRA A/B per projection = 1024*8 + 8*1024 = 16,384
4 projections/layer * 24 layers = 96 projections
HiRA A/B total = 1,572,864

model total after HiRA wrapping = 370,826,240
reported trainable count = 67,115,008
reported trainable ratio = 18.0988%
```

Why reported trainable is much larger than the effective trainable count:

```text
reported trainable = HiRA A/B + full token_embeddings + full lm_head
                   = 1,572,864 + 32,771,072 + 32,771,072
                   = 67,115,008

effective nonzero-gradient trainable = HiRA A/B + 3 special embedding rows
                                     + 3 special lm_head columns
                                   = 1,572,864 + 3,072 + 3,072
                                   = 1,579,008
```

HiRA `r=512, alpha=8` case currently visible in the local script:

```text
HiRA A/B per projection = 1024*512 + 512*1024 = 1,048,576
4 projections/layer * 24 layers = 96 projections
HiRA A/B total = 100,663,296

model total after HiRA wrapping = 469,916,672
reported trainable count = 166,205,440
reported trainable ratio ~= 35.37%

effective nonzero-gradient trainable = 100,663,296 + 3,072 + 3,072
                                   = 100,669,440
```

This explains why the printed trainable ratio can look very high even though
base vocab rows/columns are gradient-masked.

Current local HiRA implementation:

```python
class HiRALinear(nn.Module):
    def __init__(self, base, r=512, alpha=8):
        ...
        self.scale = alpha / r
        self.A = nn.Parameter(torch.randn(self.d_in, r) * 0.001)
        self.B = nn.Parameter(torch.zeros(r, self.d_out))

    def forward(self, x):
        delta = self.W_0 * (self.A @ self.B)
        ret = x @ self.W_0 + self.scale * (x @ delta)
```

Code location:

```text
sft_chat_templetes.py:263-288
```

For `r=512, alpha=8`:

```text
scale = alpha / r = 8 / 512 = 0.015625
```

So raw HiRA delta norm and effective forward delta norm are not the same.

## Current Training Data Actually Used By This SFT Function

The current local `sft()` loads:

```python
with open('./data/greetings_evol.json') as file:
    json_data = json.load(file)
```

Code location:

```text
sft_chat_templetes.py:471-474
```

The current local `sft()` resumes from:

```python
prev_ckpt_path = './checkpoints/sft_EvolSft_r_512_gpt2med_iter_30000.pt'
last_iter = modules.run_load_checkpoint(prev_ckpt_path, model, optimizer)
```

Code location:

```text
sft_chat_templetes.py:477-479
```

Important implication:

```text
The run name/checkpoint name may mention EvolSft, but the current code path is
training on ./data/greetings_evol.json after resuming the r=512 checkpoint.
```

If the remote script differs, verify these two lines first.

## Prompt Format Evidence

The training prompt formatter is:

```python
def format_chat_prompt(instruction):
    instruction = str(instruction).strip()
    return f"<|user|>\n{instruction}\n<|assistant|>\n"
```

Code location:

```text
sft_chat_templetes.py:142-144
```

The encoded training example is:

```python
prompt_tokens = tokenizer.encode(format_chat_prompt(example["instruction"]))
output_tokens = tokenizer.encode(str(example["output"]).strip())
eot_token_id = tokenizer.vocab_inv["<|endoftext|>".encode("utf-8")]
return prompt_tokens, prompt_tokens + output_tokens + [eot_token_id]
```

Code location:

```text
sft_chat_templetes.py:149-153
```

Therefore the training-side text format is:

```text
<|user|>
{instruction}
<|assistant|>
{output}<|endoftext|>
```

The user's inference prompt:

```python
inp = "<|user|>\nHello, how are you today?\n<|assistant|>\n"
```

matches the training prompt prefix exactly.

The user's inference prompt:

```python
inp = "<|user|>\nSay one word: Blue\n<|assistant|>\n"
```

also matches the training prompt prefix exactly.

Current conclusion:

```text
The latest failure is not explained by putting a space or newline in the wrong
place. The prompt format is consistent with training.
```

## Label/Shift Evidence

The batch builder creates full token sequence labels:

```python
labels = [ignore_index] * len(x_token_full)

for i in range(len(x_token_part), len(x_token_full)):
    labels[i] = x_token_full[i]
```

Code location:

```text
sft_chat_templetes.py:242-245
```

The loss function then shifts internally:

```python
logits = logits[:, :-1, :]
labels = labels[:, 1:]
```

Code location:

```text
modules.py:715-718
```

Therefore the final prompt token, i.e. the newline after `<|assistant|>`, is
trained to predict the first assistant output token. The latest observed failure
is not explained by an obvious one-token label shift bug.

## Latest User Experiments And Outputs

### Experiment 1: Greeting Prompt On Current r=512 HiRA Continuation

Prompt:

```python
inp = "<|user|>\nHello, how are you today?\n<|assistant|>\n"
```

Generation call:

```python
out = modules.generating(
    model=model,
    enc_user_prompt=inp,
    end_token=31999,
    context_len=1024,
    temperature=0.4,
    max_token=50,
    top_p=0.8
)
```

Observed output:

```text
I



"The King of the World"



 


 "The King of All"

  

  |

  I


- 

-


I
```

Training status around this output:

```text
iter 34850 loss 3.6220 lr 2.480895e-06 tokens 0.285B update_ratio 0.3926594390844305
iter 34900 loss 3.5433 lr 2.473265e-06 tokens 0.286B update_ratio 0.3926928335179885
iter 34950 loss 3.6314 lr 2.465692e-06 tokens 0.286B update_ratio 0.39272519247606397
iter 35000 loss 3.5422 lr 2.458176e-06 tokens 0.287B update_ratio 0.39275756866360706
[eval] iter 35000 | train 3.5983 | val 3.6165
assistant embedding norm tensor(3.2060)
saved checkpoint to checkpoints/sft_EvolSft_GEM_r_8_gpt2med_iter_35000.pt
```

Interpretation:

```text
At this point the model is not merely failing to be helpful. It is producing
structurally broken text with excessive blank lines, title-like fragments, and
symbol fragments.
```

### Experiment 2: Minimal Deterministic Instruction

Prompt:

```python
inp = "<|user|>\nSay one word: Blue\n<|assistant|>\n"
```

Observed current output:

```text
Touching



A


I


 


 [As    

  
 
    |


 province


Corn


ophile
```

Observed older SFT output:

```text
One word: Blue  thous

As an AI language model, I can refer you to the following words in a list:

Blue:

 clarity (blunden) ...

Pontaurozzo:
```

Interpretation:

```text
The older SFT at least latched onto "Blue" but failed the "one word" constraint
and drifted into long explanatory/list text.

The current continuation fails even this minimal instruction and emits unrelated
fragments. This is stronger evidence than the greeting prompt because the
desired behavior is unambiguous.
```

## Current update_ratio Interpretation

The user's shown `get_hira_update_ratio()` was:

```python
def get_hira_update_ratio(model):
    ratios = []
    for module in model.modules():
        if isinstance(module, HiRALinear):
            delta = module.W_0 * (module.A @ module.B)
            ratio = delta.float().norm() / module.W_0.float().norm()
            ratios.append(ratio.item())
    return sum(ratios) / len(ratios)
```

This measures raw:

```text
norm(W_0 * (A @ B)) / norm(W_0)
```

But HiRA forward uses:

```text
x @ W_0 + scale * x @ (W_0 * (A @ B))
```

So the effective forward ratio is:

```text
effective_update_ratio = raw_update_ratio * alpha / r
```

For the latest clarified first segment:

```text
r = 512
alpha = 8
scale = 0.015625
```

Therefore:

```text
raw update_ratio 15 -> effective ratio about 0.234
raw update_ratio 19 -> effective ratio about 0.297
```

The current local code appears to have `get_hira_update_ratio()` already changed
to multiply `module.scale`, but verify the remote script. If the remote script
still uses the raw version above, its printed ratio must be multiplied by
`alpha / r` before comparing across runs.

Code locations:

```text
sft_chat_templetes.py:34-41
sft_chat_templetes.py:263-288
```

## Generation Code Evidence

The generation function defaults to sampling:

```python
do_sample: bool = True
temperature: float = 0.8
top_p: float = 0.9
repetition_penalty: float = 1.1
no_repeat_ngram_size: int = 4
```

Code location:

```text
modules.py:1210-1222
```

During generation, the repetition penalty is applied to:

```python
generated_ids = enc_user_prompt[0].tolist()
output_logits = apply_repetition_penalty(
    output_logits,
    generated_ids,
    penalty=repetition_penalty,
)
```

Code location:

```text
modules.py:1236-1244
```

This means the repetition penalty applies to the entire prompt, not only to
newly generated assistant tokens.

Implication for the test:

```text
For prompt "Say one word: Blue", the correct answer token "Blue" is already in
the prompt, so the default repetition penalty can suppress the correct answer.
This can worsen the one-word test, but it cannot fully explain unrelated output
such as "Touching ... province ... Corn ... ophile".
```

The generation function also applies no-repeat-ngram masking over the full prompt
plus generated sequence:

```python
output_logits = apply_no_repeat_ngram_mask(
    output_logits,
    generated_ids,
    ngram_size=no_repeat_ngram_size,
)
```

Code location:

```text
modules.py:1245-1249
```

Recommended diagnostic generation call:

```python
out = modules.generating(
    model=model,
    enc_user_prompt=inp,
    end_token=31999,
    context_len=1024,
    max_token=10,
    do_sample=False,
    repetition_penalty=1.0,
    no_repeat_ngram_size=None,
    ban_token_ids=[32000, 32001, 32002],
)
print(tk.decode(out))
```

This separates model distribution failure from sampling/repetition-penalty
artifacts.

## Current Main Inference

The latest failures are best explained as:

```text
The current SFT checkpoint has not learned reliable instruction following and
its assistant-start distribution is degraded. It is not primarily a chat prompt
format problem.
```

Supporting points:

1. Training and inference prompt prefixes match exactly.
2. The label shift path is internally consistent.
3. The minimal instruction `Say one word: Blue` fails badly.
4. The older SFT recognized `Blue` but still ignored the one-word constraint,
   showing weak instruction following rather than a pure tokenization issue.
5. The current continuation emits unrelated fragments, suggesting a degraded
   generation distribution.
6. The current run is resuming a HiRA SFT checkpoint and then training on
   `greetings_evol.json`; if that is also true remotely, the data mixture is too
   narrow/small to fix general instruction following and may overfit style.
7. The generation defaults can make tests worse because repetition/no-repeat
   penalties are applied to the full prompt, but they are not sufficient to
   explain the whole failure.

## Practical Read Of The Current Checkpoint

Given:

```text
val loss around 3.6
failure on one-word instruction
fragmentary sampled output
effective/raw update ratio needs careful interpretation by r/alpha
```

The current checkpoint should not be judged healthy based only on decreasing
loss. The latest evidence says it is not yet a usable instruction-following
checkpoint.

Before making further training decisions, use the deterministic diagnostic above
and inspect the first-token top-k distribution after:

```text
<|user|>
Say one word: Blue
<|assistant|>
```

If `Blue` is not near the top under greedy/no-penalty conditions, the problem is
training/data/checkpoint quality, not generation sampling.
