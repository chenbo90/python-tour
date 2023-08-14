from transformers import GPT2LMHeadModel, GPT2Tokenizer, BertForSequenceClassification, BertTokenizer

# 加载GPT-2模型和分词器
gpt2_model_name = "gpt2"
gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_model_name)
gpt2_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name)

# 加载BERT模型和分词器
bert_model_name = "bert-base-uncased"
bert_model = BertForSequenceClassification.from_pretrained(bert_model_name)
print(bert_model)
bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)

# 从GPT-2模型中提取权重
gpt2_weights = gpt2_model.state_dict()

# 将GPT-2权重转移到BERT模型的对应层
bert_model.bert.embeddings.word_embeddings.weight.data = gpt2_weights["transformer.wte.weight"]

print(bert_model)

# 可能需要进一步调整其他权重的形状和名称，以适应BERT模型的结构

# 在新数据上进行微调（假设有适当的文本分类数据集）
# ...

# 进行测试和验证
# ...
