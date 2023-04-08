from transformers import AutoTokenizer, AutoModelForMaskedLM, DebertaForMaskedLM, DebertaTokenizer
import torch
#
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")
model = DebertaForMaskedLM.from_pretrained("microsoft/deberta-base")

# tokenizer = DebertaTokenizer.from_pretrained("lsanochkin/deberta-large-feedback")
# model = DebertaForMaskedLM.from_pretrained("lsanochkin/deberta-large-feedback")


text = "I love this movie. The sentiment of this sentence is [MASK]."
inputs = tokenizer(text, return_tensors='pt')
with torch.no_grad():
    logits = model(**inputs).logits

mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
# get the probability of good and bad for mask token index
mask_token_logits = logits[0, mask_token_index, :].squeeze()

# verbalizer = ['Paris', 'Washington', 'London', 'Tokyo', 'Beijing']
verbalizer = ['\u0120terrible', '\u0120great']
# get the probability of the tokens in the verbalizer
probs = torch.softmax(mask_token_logits[tokenizer.convert_tokens_to_ids(verbalizer)], dim=-1)
probs