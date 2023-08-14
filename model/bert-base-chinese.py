import torch
from transformers import BertModel, BertTokenizer,BertConfig,AutoConfig,AutoModel,AutoTokenizer
# 首先要import进来

print("====方式1:手动指定模型类别====")
config1 = BertConfig.from_pretrained('bert-base-chinese')

model1 = BertModel.from_pretrained("bert-base-chinese")
#可以直接更改模型配置
update_config = config1.update({'output_hidden_states': True})
model11 = BertModel.from_pretrained("bert-base-chinese",config=update_config)

#根据config配置来初始化
model12 = BertModel(config1)

print(model1)

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
print(tokenizer.encode("生活的真谛是美和爱"))  # 对于单个句子编码



print("====方式2：通用型接口，根据模型名称自动载入和初始化不同的预训练模型====")
config2 = AutoConfig.from_pretrained('bert-base-chinese')

#1:from_pretrained()方法
mode21 = AutoModel.from_pretrained("bert-base-chinese")
mode22 = AutoModel.from_pretrained("bert-base-chinese", config2)

#2:from_config()方法
model23 = AutoModel.from_config(config2)
print(mode21)

#分词
tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
print(tokenizer.encode("生活的真谛是美和爱"))  # 对于单个句子编码

print("====方式3：根据config配置创建模型====")
config3 = BertConfig.from_pretrained('bert-base-chinese')
model3 = BertModel(config3)

# print()



'''
BertModel(
  (embeddings): BertEmbeddings(
    (word_embeddings): Embedding(21128, 768, padding_idx=0)
    (position_embeddings): Embedding(512, 768)
    (token_type_embeddings): Embedding(2, 768)
    (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
    (dropout): Dropout(p=0.1, inplace=False)
  )
  (encoder): BertEncoder(
    (layer): ModuleList(
      (0-11): 12 x BertLayer(
        (attention): BertAttention(
          (self): BertSelfAttention(
            (query): Linear(in_features=768, out_features=768, bias=True)
            (key): Linear(in_features=768, out_features=768, bias=True)
            (value): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (output): BertSelfOutput(
            (dense): Linear(in_features=768, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (intermediate): BertIntermediate(
          (dense): Linear(in_features=768, out_features=3072, bias=True)
          (intermediate_act_fn): GELUActivation()
        )
        (output): BertOutput(
          (dense): Linear(in_features=3072, out_features=768, bias=True)
          (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
    )
  )
  (pooler): BertPooler(
    (dense): Linear(in_features=768, out_features=768, bias=True)
    (activation): Tanh()
  )
)
'''







print(tokenizer.encode_plus("生活的真谛是美和爱","说的太好了")) # 对于一组句子编码