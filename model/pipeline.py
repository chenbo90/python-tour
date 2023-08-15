from transformers import pipeline
# 运行该段代码要保障你的电脑能够上网，会自动下载预训练模型，大概420M
unmasker = pipeline("fill-mask",model = "bert-base-uncased")  # 这里引入了一个任务叫fill-mask，该任务使用了base的bert模型
l = unmasker("The goal of life is [MASK].", top_k=5)
print(l)

translator = pipeline("translation_en_to_fr")
print(translator("How old are you?"))


