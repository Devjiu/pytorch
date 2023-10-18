import transformers

import torch
import torch.nn.functional as F
from torch import nn

from datasets import load_dataset
from tqdm.auto import tqdm
import torch.optim as optim

import intel_extension_for_pytorch as ipex

# ipex.enable_auto_mixed_precision(mixed_dtype = torch.bfloat16)

model_name = "EleutherAI/gpt-j-6B"
gpt = transformers.AutoModelForCausalLM.from_pretrained(model_name)

device = torch.device("cpu")

tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
prompt = tokenizer("A cat sat on a mat", return_tensors='pt')
prompt = {key: value.to(device) for key, value in prompt.items()}
out = gpt.generate(**prompt, min_length=128, max_length=128, do_sample=True)
print(tokenizer.decode(out[0]))

# model.gradient_checkpointing_enable()
tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
codeparrot = load_dataset("transformersbook/codeparrot-train", streaming=True)
optimizer = optim.Adam(gpt.parameters(), lr=1e-5)
gpt.train()
gpt, optimizer = ipex.optimize(gpt, optimizer=optimizer)

# with torch.cuda.amp.autocast():
for row in tqdm(codeparrot["train"]):
    if len(row["content"]) <= 1:
        continue

    batch = tokenizer(row["content"], truncation=True, max_length=128, return_tensors='pt')
    # print(batch)
    batch = {k: v.to(device) for k, v in batch.items()}
    # print(batch["input_ids"].shape)
    # print(batch["attention_mask"].shape)

    out = gpt.forward(**batch)
    # out = model.forward(batch["input_ids"], batch["attention_mask"])

    loss = F.cross_entropy(out.logits[:, :-1, :].flatten(0, -2), batch['input_ids'][:, 1:].flatten(),
                           reduction='mean')
    print(loss)
    loss.backward()

    optimizer.step()
    optimizer.zero_grad()