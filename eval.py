# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import dotenv
import os
import torch

dotenv.load_dotenv()

HF_ACCESS_TOKEN = os.getenv("HF_ACCESS_TOKEN")

# login
from huggingface_hub import login
login(HF_ACCESS_TOKEN)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
messages = [
	{"role": "user", "content": "Who are you?"},
]
inputs = tokenizer.apply_chat_template(
	messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
).to(model.device)

# outputs = model.generate(**inputs, max_new_tokens=40)
# print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))

def log_likelihood(text):
	# Tokenize with special tokens if needed
	inputs = tokenizer(text, return_tensors="pt")
	input_ids = inputs["input_ids"]
	attention_mask = inputs["attention_mask"]

	with torch.no_grad():
		outputs = model(input_ids, attention_mask=attention_mask)
		logits = outputs.logits
		print(logits)

	# Shift logits and labels so that tokens predict the next token
	shift_logits = logits[:, :-1, :].contiguous()
	shift_labels = input_ids[:, 1:].contiguous()

	# Compute log-softmax over vocab to get log-probs
	log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)

	# Gather log-probabilities for the actual next tokens
	token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

	# Sum over all tokens to get total log-likelihood
	total_log_likelihood = token_log_probs.sum(dim=1)
	return total_log_likelihood.item(), token_log_probs.squeeze(0).tolist()

text = "The quick brown fox jumps over the lazy dog."
total_ll, token_lls = log_likelihood(text)
print("Total log likelihood:", total_ll)
print("Per-token log likelihoods:", token_lls)
