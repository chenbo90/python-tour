from transformers import AutoModel,TFAutoModel,FlaxAutoModel, AutoTokenizer

# 选择框架：pytorch、tensorflow、jax
framework = "pytorch"  # 或者 "tensorflow"、"jax"

# 预训练模型的名称
model_name = "bert-base-uncased"

# 加载模型和分词器
if framework == "pytorch":
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
elif framework == "tensorflow":
    model = TFAutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
elif framework == "jax":
    model = FlaxAutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
else:
    raise ValueError("Unsupported framework")

# 示例输入文本
input_text = "Hugging Face's transformers library is versatile!"

# 使用分词器对文本进行编码
inputs = tokenizer(input_text, return_tensors=framework)

# 使用模型进行推理
outputs = model(**inputs)

# 输出模型的结果
print(outputs)
