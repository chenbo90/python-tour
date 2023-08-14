# 下面给出bert-base-uncased的例子，实现对两个句子相似度的计算

# 导入tokenizer
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

input = tokenizer('The first sentence!', 'The second sentence!')
# print(input)
'''
{'input_ids': [101, 1996, 2034, 6251, 999, 102, 1996, 2117, 6251, 999, 102], 'token_type_ids': [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
'''
# print(tokenizer.convert_ids_to_tokens(input["input_ids"]))
# output
'''
['[CLS]', 'the', 'first', 'sentence', '!', '[SEP]', 'the', 'second', 'sentence', '!', '[SEP]']
'''


# 实际使用tokenizer的方法，得到tokenizer_data
def tokenize_function(example):
    return tokenizer(example["sentence1"], example['sentence2'], truncation=True)


from datasets import load_dataset

datasets = load_dataset("glue", "mrpc")
tokenizer_data = datasets.map(tokenize_function, batched=True)
# print(tokenizer_data)

# 训练参数
from transformers import TrainingArguments

trainging_arg = TrainingArguments("test-trainer")
# print(trainging_arg)  # 看下默认值
'''
TrainingArguments(
_n_gpu=0,
adafactor=False,
adam_beta1=0.9,
adam_beta2=0.999,
adam_epsilon=1e-08,
auto_find_batch_size=False,
bf16=False,
bf16_full_eval=False,
data_seed=None,
dataloader_drop_last=False,
dataloader_num_workers=0,
dataloader_pin_memory=True,
ddp_backend=None,
ddp_broadcast_buffers=None,
ddp_bucket_cap_mb=None,
ddp_find_unused_parameters=None,
ddp_timeout=1800,
debug=[],
deepspeed=None,
disable_tqdm=False,
do_eval=False,
do_predict=False,
do_train=False,
eval_accumulation_steps=None,
eval_delay=0,
eval_steps=None,
evaluation_strategy=no,
fp16=False,
fp16_backend=auto,
fp16_full_eval=False,
fp16_opt_level=O1,
fsdp=[],
fsdp_config={'fsdp_min_num_params': 0, 'xla': False, 'xla_fsdp_grad_ckpt': False},
fsdp_min_num_params=0,
fsdp_transformer_layer_cls_to_wrap=None,
full_determinism=False,
gradient_accumulation_steps=1,
gradient_checkpointing=False,
greater_is_better=None,
group_by_length=False,
half_precision_backend=auto,
hub_model_id=None,
hub_private_repo=False,
hub_strategy=every_save,
hub_token=<HUB_TOKEN>,
ignore_data_skip=False,
include_inputs_for_metrics=False,
jit_mode_eval=False,
label_names=None,
label_smoothing_factor=0.0,
learning_rate=5e-05,
length_column_name=length,
load_best_model_at_end=False,
local_rank=0,
log_level=passive,
log_level_replica=warning,
log_on_each_node=True,
logging_dir=test-trainer/runs/Aug11_14-27-49_node-10-21-10-205,
logging_first_step=False,
logging_nan_inf_filter=True,
logging_steps=500,
logging_strategy=steps,
lr_scheduler_type=linear,
max_grad_norm=1.0,
max_steps=-1,
metric_for_best_model=None,
mp_parameters=,
no_cuda=False,
num_train_epochs=3.0,
optim=adamw_hf,
optim_args=None,
output_dir=test-trainer,
overwrite_output_dir=False,
past_index=-1,
per_device_eval_batch_size=8,
per_device_train_batch_size=8,
prediction_loss_only=False,
push_to_hub=False,
push_to_hub_model_id=None,
push_to_hub_organization=None,
push_to_hub_token=<PUSH_TO_HUB_TOKEN>,
ray_scope=last,
remove_unused_columns=True,
report_to=['tensorboard'],
resume_from_checkpoint=None,
run_name=test-trainer,
save_on_each_node=False,
save_safetensors=False,
save_steps=500,
save_strategy=steps,
save_total_limit=None,
seed=42,
sharded_ddp=[],
skip_memory_metrics=True,
tf32=None,
torch_compile=False,
torch_compile_backend=None,
torch_compile_mode=None,
torchdynamo=None,
tpu_metrics_debug=False,
tpu_num_cores=None,
use_ipex=False,
use_legacy_prediction_loop=False,
use_mps_device=False,
warmup_ratio=0.0,
warmup_steps=0,
weight_decay=0.0,
xpu_backend=None,
)
'''

# 导入模型
from transformers import AutoModelForSequenceClassification,AutoModel

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased",num_labels=2)

from transformers import DataCollatorWithPadding 

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
#
from transformers import Trainer

trainer = Trainer(model, trainging_arg, train_dataset=tokenizer_data['train'],
                  eval_dataset=tokenizer_data["validation"], data_collator=data_collator, tokenizer=tokenizer)

train = trainer.train()
print(train)
