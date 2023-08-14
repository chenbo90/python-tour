from transformers import GPT2LMHeadModel, GPT2Tokenizer, BertForSequenceClassification, BertTokenizer

# 加载GPT-2模型和分词器
gpt2_model_name = "gpt2"
gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_model_name)
gpt2_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name)

# 加载BERT模型和分词器
bert_model_name = "bert-base-uncased"
bert_model = BertForSequenceClassification.from_pretrained(bert_model_name)
print(bert_model.bert.embeddings.word_embeddings.weight.data)
'''
tensor([[-0.0102, -0.0615, -0.0265,  ..., -0.0199, -0.0372, -0.0098],
        [-0.0117, -0.0600, -0.0323,  ..., -0.0168, -0.0401, -0.0107],
        [-0.0198, -0.0627, -0.0326,  ..., -0.0165, -0.0420, -0.0032],
        ...,
        [-0.0218, -0.0556, -0.0135,  ..., -0.0043, -0.0151, -0.0249],
        [-0.0462, -0.0565, -0.0019,  ...,  0.0157, -0.0139, -0.0095],
        [ 0.0015, -0.0821, -0.0160,  ..., -0.0081, -0.0475,  0.0753]])

'''
bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)

# 从GPT-2模型中提取权重
gpt2_weights = gpt2_model.state_dict()

# 将GPT-2权重转移到BERT模型的对应层
bert_model.bert.embeddings.word_embeddings.weight.data = gpt2_weights["transformer.wte.weight"]



print(bert_model.bert.embeddings.word_embeddings.weight.data)
'''
tensor([[-0.1101, -0.0393,  0.0331,  ..., -0.1364,  0.0151,  0.0453],
        [ 0.0403, -0.0486,  0.0462,  ...,  0.0861,  0.0025,  0.0432],
        [-0.1275,  0.0479,  0.1841,  ...,  0.0899, -0.1297, -0.0879],
        ...,
        [-0.0445, -0.0548,  0.0123,  ...,  0.1044,  0.0978, -0.0695],
        [ 0.1860,  0.0167,  0.0461,  ..., -0.0963,  0.0785, -0.0225],
        [ 0.0514, -0.0277,  0.0499,  ...,  0.0070,  0.1552,  0.1207]])
'''

# 可能需要进一步调整其他权重的形状和名称，以适应BERT模型的结构

# 在新数据上进行微调（假设有适当的文本分类数据集）
# ...

# 进行测试和验证
# ...
