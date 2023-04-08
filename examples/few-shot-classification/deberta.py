from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
model = AutoModelForMaskedLM.from_pretrained("microsoft/deberta-v3-base")




text = "The capital of France is [MASK]."
inputs = tokenizer(text, return_tensors='pt')
with torch.no_grad():
    logits = model(**inputs).logits

mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
# get the probability of good and bad for mask token index
mask_token_logits = logits[0, mask_token_index, :].squeeze()

verbalizer = ['Paris', 'Washington', 'London', 'Tokyo', 'Beijing']
# get the probability of the tokens in the verbalizer
probs = torch.softmax(mask_token_logits[tokenizer.convert_tokens_to_ids(verbalizer)], dim=-1)
probs

# get the word by token id
tokenizer.convert_ids_to_tokens(torch.argmax(mask_token_logits).item())