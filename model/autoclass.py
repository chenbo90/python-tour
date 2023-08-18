import transformers

for api in dir(transformers):
    if api.startswith('AutoModelFor'):
        print(api)
'''
AutoModelForAudioClassification
AutoModelForAudioFrameClassification
AutoModelForAudioXVector
AutoModelForCTC
AutoModelForCausalLM
AutoModelForDepthEstimation
AutoModelForDocumentQuestionAnswering
AutoModelForImageClassification
AutoModelForImageSegmentation
AutoModelForInstanceSegmentation
AutoModelForMaskedImageModeling
AutoModelForMaskedLM
AutoModelForMultipleChoice
AutoModelForNextSentencePrediction
AutoModelForObjectDetection
AutoModelForPreTraining
AutoModelForQuestionAnswering
AutoModelForSemanticSegmentation
AutoModelForSeq2SeqLM
AutoModelForSequenceClassification
AutoModelForSpeechSeq2Seq
AutoModelForTableQuestionAnswering
AutoModelForTokenClassification
AutoModelForVideoClassification
AutoModelForVision2Seq
AutoModelForVisualQuestionAnswering
AutoModelForZeroShotObjectDetection

'''