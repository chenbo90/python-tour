# from all_in_one import test_loop
import all_in_one as ai
import torch
from transformers import AutoConfig,BertForPairwiseCLS,AutoTokenizer
checkpoint = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

# config = AutoConfig.from_pretrained(checkpoint)
# model = BertForPairwiseCLS.from_pretrained(checkpoint, config=config).to(device)

ai.model.load_state_dict(torch.load('epoch_3_valid_acc_74.1_model_weights.bin'))
ai.test_loop(valid_dataloader, ai.model, mode='Test')