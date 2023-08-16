# download from Use in Transformers
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")

# 分词
'''
return_tensors是一个参数，用于指定在使用分词器进行编码时，返回的结果的张量类型。
它可以用于将结果直接返回为PyTorch张量或TensorFlow张量，以便进行后续的深度学习计算。
'''
inputs = tokenizer("Learning is a very happy [MASK].", return_tensors='pt')
print(inputs)
'''
{'input_ids': tensor([[ 101, 4083, 2003, 1037, 2200, 3407,  103, 1012,  102]]), 'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1]])}
'''

# 然后将第一步的结果作为模型的入参
argmax = model(**inputs).logits.argmax(dim=-1)
'''
tensor([[1012, 4083, 2003, 1037, 2200, 3407, 2832, 1012, 1012]])
'''

# 转换id到词
tokens = tokenizer.convert_ids_to_tokens(2832)
print(tokens)
'''
'process'    #这里我们得到了和页面同样的数据
'''

print(model.bert)
outputs = model.bert(**inputs)
print(outputs)
print(outputs.last_hidden_state.size())

import torch
from torch import nn  # 定义最后的二分类线性层

cls = nn.Sequential(nn.Linear(768, 1), nn.Sigmoid()
                    )
# 使用二分类常用的Binary Cross Entropy Loss
criteria = nn.BCELoss()
# 这里只对最后的线性层做参数更新
optimizer = torch.optim.SGD(cls.parameters(), lr=0.1)
# 取隐层的第一个token(<bos>)的输出作为cls层的输入，然后与label进行损失计算
loss = criteria(cls(outputs.last_hidden_state[:, 0, :]), torch.FloatTensor([[1]]))
loss.backward()
optimizer.step()
optimizer.zero_grad()
