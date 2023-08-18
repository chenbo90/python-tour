import all_in_one as ai
import torch


load = torch.load('epoch_3_valid_acc_73.9_model_weights.bin')
print(load)
state_dict = ai.model.load_state_dict(load)
print(state_dict)
ai.test_loop(ai.valid_dataloader, ai.model, mode='Test')