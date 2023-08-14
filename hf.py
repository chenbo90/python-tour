from transformers import pipeline

classifier = pipeline("sentiment-analysis")
print("------")
print(classifier("I hava been waiting for a HuggingFace course my whole life"))

def analysis(str):
   return classifier(str)
# if __name__ == '__main__':
#     classifier = pipeline("sentiment-analysis")
#     classifier("I hava been waiting for a HuggingFace course my whole life")
