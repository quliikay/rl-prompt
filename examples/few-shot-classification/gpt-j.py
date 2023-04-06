import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-j-6B', revision="float16", torch_dtype=torch.float16, pad_token='<|endoftext|>')
model = AutoModelForCausalLM.from_pretrained('EleutherAI/gpt-j-6B', revision="float16", torch_dtype=torch.float16).to(device)

text = 'I like this movie . The sentiment of this sentence is '
input_ids = tokenizer(text, return_tensors='pt', padding='longest', truncation=True)

logits = model(**input_ids.to(device)).logits
# get top 5 tokens with highest probability
probs = torch.softmax(logits[0, -1, :], dim=-1)
top5 = torch.topk(probs, 5)[1]
print(tokenizer.decode(top5))

# get the word of token that id is 220
print(tokenizer.decode(220))

next_token = torch.argmax(logits[0, -1, :])
next_token = tokenizer.decode(next_token)
verbalizer = ['\u0120terrible', '\u0120great']
probs = torch.softmax(logits[0, -1, tokenizer.convert_tokens_to_ids(verbalizer)], dim=-1)
print(probs)