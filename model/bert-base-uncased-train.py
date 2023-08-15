#download from Use in Transformers
from transformers import AutoTokenizer, AutoModelForMaskedLM
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")

#分词
inputs = tokenizer("Learning is a very happy [MASK].", return_tensors='pt')
print(inputs)
'''
{'input_ids': tensor([[ 101, 4083, 2003, 1037, 2200, 3407,  103, 1012,  102]]), 'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1]])}
'''

#然后将第一步的结果作为模型的入参
argmax = model(**inputs).logits.argmax(dim=-1)
'''
tensor([[1012, 4083, 2003, 1037, 2200, 3407, 2832, 1012, 1012]])
'''

#转换id到词
tokens = tokenizer.convert_ids_to_tokens(2832)
print(tokens)
'''
'process'    #这里我们得到了和页面同样的数据
'''

