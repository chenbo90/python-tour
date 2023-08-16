import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


# 定义自定义模型
class CustomModel(nn.Module):
    def __init__(self, num_classes):
        super(CustomModel, self).__init__()
        self.bert = AutoModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        logits = self.fc(pooled_output)
        return logits


# 加载预训练tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# 示例输入
text = "This is a sample sentence for classification."
input_ids = tokenizer.encode(text, add_special_tokens=True)
attention_mask = [1] * len(input_ids)

# 初始化自定义模型
num_classes = 2  # 根据你的任务设定分类数量
model = CustomModel(num_classes)

# 将输入转换为PyTorch张量
input_ids = torch.tensor(input_ids).unsqueeze(0)  # 添加批次维度
attention_mask = torch.tensor(attention_mask).unsqueeze(0)  # 添加批次维度

# 前向传播
logits = model(input_ids, attention_mask)

# 打印预测结果
predicted_class = torch.argmax(logits, dim=1).item()
print("Predicted Class:", predicted_class) #0-积极 1-消极
