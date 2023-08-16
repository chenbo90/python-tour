import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class SentimentAnalysisModel(nn.Module):
    def __init__(self, num_classes, model_name='bert-base-uncased'):
        super(SentimentAnalysisModel, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.fc(pooled_output)
        return logits


# 加载预训练tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# 示例输入
texts = ["I love this movie!", "This is a bad product."]
input_ids = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')['input_ids']
attention_mask = input_ids != tokenizer.pad_token_id

# 初始化自定义模型
num_classes = 2  # 积极/消极
model = SentimentAnalysisModel(num_classes)

# 前向传播
logits = model(input_ids, attention_mask)

# 预测结果
predicted_classes = torch.argmax(logits, dim=1)
print("Predicted Classes:", predicted_classes)
