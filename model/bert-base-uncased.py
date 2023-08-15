from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokenizer1 = tokenizer("I'm learning deep learning.")
print(tokenizer1)