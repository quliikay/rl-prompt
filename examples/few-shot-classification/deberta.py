from transformers import \
    AutoTokenizer, AutoModelForMaskedLM, DebertaForMaskedLM, \
    DebertaTokenizer, BertForMaskedLM, AutoModelForCausalLM, GPT2LMHeadModel
import torch
#
# tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")
# model = DebertaForMaskedLM.from_pretrained("microsoft/deberta-base")

tokenizer = DebertaTokenizer.from_pretrained("lsanochkin/deberta-large-feedback")
model = DebertaForMaskedLM.from_pretrained("lsanochkin/deberta-large-feedback")

# tokenizer = AutoTokenizer.from_pretrained("bert-large-cased")
# model = BertForMaskedLM.from_pretrained("bert-large-cased")

# tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-j-6B', pad_token='<|endoftext|>', revision="float16", torch_dtype=torch.float16)
# model = (AutoModelForCausalLM.from_pretrained(
#                 'EleutherAI/gpt-j-6B', revision="float16", torch_dtype=torch.float16,
#             ))

# tokenizer = AutoTokenizer.from_pretrained('gpt2-large', pad_token='<|endoftext|>')
# model = GPT2LMHeadModel.from_pretrained('gpt2-large')
model.config.pad_token_id = tokenizer.pad_token_id

text = "I don't like this movie. The sentiment of this sentence is [MASK]."
inputs = tokenizer(text, return_tensors='pt')
with torch.no_grad():
    logits = model(**inputs).logits

mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
# get the probability of good and bad for mask token index
mask_token_logits = logits[0, mask_token_index, :].squeeze()

# verbalizer = ['Paris', 'Washington', 'London', 'Tokyo', 'Beijing']
verbalizer = ['terrible', 'great']
# get the probability of the tokens in the verbalizer
probs = torch.softmax(mask_token_logits[tokenizer.convert_tokens_to_ids(verbalizer)], dim=-1)
probs