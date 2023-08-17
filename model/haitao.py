# from transformers import pipeline, set_seed
# generator = pipeline('text-generation', model='HaiTao90/gpt2-wiki')
# set_seed(42)
# out = generator("Hello, I'm a language model,", max_length=30, num_return_sequences=5)
# print(out)


from transformers import GPT2Tokenizer, GPT2Model, AutoModel, AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained('HaiTao90/gpt2-wiki')
model = AutoModelForCausalLM.from_pretrained('HaiTao90/gpt2-wiki')
#model = AutoModel.from_pretrained('HaiTao90/gpt2-wiki')
text = "Replace me by any text you'd like."
# encoded_input = tokenizer(text, return_tensors='pt')
# output = model(**encoded_input)
#
# print(output)
# logits = output[1]["data"]
# print(logits)

# encode context the generation is conditioned on
model_inputs = tokenizer('I enjoy walking with my cute dog', return_tensors='pt').to("cpu")

# generate 40 new tokens
greedy_output = model.generate(**model_inputs, max_new_tokens=40)

print("Output:\n" + 100 * '-')
print(tokenizer.decode(greedy_output[0], skip_special_tokens=True))