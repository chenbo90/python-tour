#来源：https://huggingface.co/learn/nlp-course/chapter7/3?fw=pt
from transformers import AutoModelForMaskedLM

model_checkpoint = "distilbert-base-uncased"
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)

distilbert_num_parameters = model.num_parameters() / 1_000_000
print(f"'>>> DistilBERT number of parameters: {round(distilbert_num_parameters)}M'")
print(f"'>>> BERT number of parameters: 110M'")

text = "This is a great [MASK]."

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


import torch

inputs = tokenizer(text, return_tensors="pt")
token_logits = model(**inputs).logits
# Find the location of [MASK] and extract its logits
mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
mask_token_logits = token_logits[0, mask_token_index, :]
# Pick the [MASK] candidates with the highest logits
top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()

for token in top_5_tokens:
    print(f"'>>> {text.replace(tokenizer.mask_token, tokenizer.decode([token]))}'")

'''
output:
'>>> This is a great deal.'
'>>> This is a great success.'
'>>> This is a great adventure.'
'>>> This is a great idea.'
'>>> This is a great feat.'
'''

from datasets import load_dataset

#第一种方式：从hub上传下载
#imdb_dataset = load_dataset("imdb")

#第二种方式，从本地缓存加载
#205环境
#imdb_dataset = load_dataset("imdb",cache_dir='/home/ubuntu/.cache/huggingface/datasets')

#211环境
imdb_dataset = load_dataset("imdb",cache_dir='/root/.cache/huggingface/datasets')


print(imdb_dataset)

'''
output:
DatasetDict({
    train: Dataset({
        features: ['text', 'label'],
        num_rows: 25000
    })
    test: Dataset({
        features: ['text', 'label'],
        num_rows: 25000
    })
    unsupervised: Dataset({
        features: ['text', 'label'],
        num_rows: 50000
    })
})
'''


'''
随机打印看看
seed：随机种子，这可以使随机性的行为在不同运行中保持一致
'''
sample = imdb_dataset["train"].shuffle(seed=42).select(range(3))

for row in sample:
    print(f"\n'>>> Review: {row['text']}'")
    print(f"'>>> Label: {row['label']}'")

'''
out:
'>>> Review: There is no relation at all between Fortier and Profiler but the fact that both are police series about violent crimes. Profiler looks crispy, Fortier looks classic. Profiler plots are quite simple. Fortier's plot are far more complicated... Fortier looks more like Prime Suspect, if we have to spot similarities... The main character is weak and weirdo, but have "clairvoyance". People like to compare, to judge, to evaluate. How about just enjoying? Funny thing too, people writing Fortier looks American but, on the other hand, arguing they prefer American series (!!!). Maybe it's the language, or the spirit, but I think this series is more English than American. By the way, the actors are really good and funny. The acting is not superficial at all...'
'>>> Label: 1'

'>>> Review: This movie is a great. The plot is very true to the book which is a classic written by Mark Twain. The movie starts of with a scene where Hank sings a song with a bunch of kids called "when you stub your toe on the moon" It reminds me of Sinatra's song High Hopes, it is fun and inspirational. The Music is great throughout and my favorite song is sung by the King, Hank (bing Crosby) and Sir "Saggy" Sagamore. OVerall a great family movie or even a great Date movie. This is a movie you can watch over and over again. The princess played by Rhonda Fleming is gorgeous. I love this movie!! If you liked Danny Kaye in the Court Jester then you will definitely like this movie.'
'>>> Label: 1'

'>>> Review: George P. Cosmatos' "Rambo: First Blood Part II" is pure wish-fulfillment. The United States clearly didn't win the war in Vietnam. They caused damage to this country beyond the imaginable and this movie continues the fairy story of the oh-so innocent soldiers. The only bad guys were the leaders of the nation, who made this war happen. The character of Rambo is perfect to notice this. He is extremely patriotic, bemoans that US-Americans didn't appreciate and celebrate the achievements of the single soldier, but has nothing but distrust for leading officers and politicians. Like every film that defends the war (e.g. "We Were Soldiers") also this one avoids the need to give a comprehensible reason for the engagement in South Asia. And for that matter also the reason for every single US-American soldier that was there. Instead, Rambo gets to take revenge for the wounds of a whole nation. It would have been better to work on how to deal with the memories, rather than suppressing them. "Do we get to win this time?" Yes, you do.'
'>>> Label: 0'
'''

def tokenize_function(examples):
    result = tokenizer(examples["text"])
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    return result


# Use batched=True to activate fast multithreading!
tokenized_datasets = imdb_dataset.map(
    tokenize_function, batched=True, remove_columns=["text", "label"]
)
print(tokenized_datasets)

'''
out:
DatasetDict({                                                                                                                  
    train: Dataset({
        features: ['input_ids', 'attention_mask', 'word_ids'],
        num_rows: 25000
    })
    test: Dataset({
        features: ['input_ids', 'attention_mask', 'word_ids'],
        num_rows: 25000
    })
    unsupervised: Dataset({
        features: ['input_ids', 'attention_mask', 'word_ids'],
        num_rows: 50000
    })
})
'''

#检查标记器的model_max_length 属性，
length = tokenizer.model_max_length
print(length)

chunk_size = 128

# Slicing produces a list of lists for each feature
tokenized_samples = tokenized_datasets["train"][:3]

for idx, sample in enumerate(tokenized_samples["input_ids"]):
    print(f"'>>> Review {idx} length: {len(sample)}'")

def group_texts(examples): # chunk to 128 dim

    # Concatenate all texts
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}

    # Compute length of concatenated texts
    total_length = len(concatenated_examples[list(examples.keys())[0]])

    # We drop the last chunk if it's smaller than chunk_size
    total_length = (total_length // chunk_size) * chunk_size

    # Split by chunks of max_len
    result = {
        k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }
    # Create a new labels column
    result["labels"] = result["input_ids"].copy()
    return result

lm_datasets = tokenized_datasets.map(group_texts, batched=True)


from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

def insert_random_mask(batch):
    features = [dict(zip(batch, t)) for t in zip(*batch.values())] #这个 features 是一个list，里面有 batch_sizes个字典, keys: 'input_ids', 'attention_mask', 'labels'
    masked_inputs = data_collator(features) # 列表转化为 字典, 同时加random mask
    # Create a new "masked" column for each column in the dataset
    return {"masked_" + k: v.numpy() for k, v in masked_inputs.items()} # 字典的keys加上 "masked" + 'input_ids', 'attention_mask', 'labels'

batch_size = 12
downsampled_dataset = lm_datasets["train"].train_test_split(train_size=1000, test_size=100, seed=42)
downsampled_dataset = downsampled_dataset.remove_columns(["word_ids"])

train_dataloader = DataLoader(
    downsampled_dataset["train"],
    shuffle=True,
    batch_size=batch_size,
    collate_fn=data_collator,
)

for batch in train_dataloader:
    y2=insert_random_mask(batch)
    print(y2)
    break

import collections
from transformers import default_data_collator

wwm_probability = 0.2

def whole_word_masking_data_collator(features):
    for feature in features:
        word_ids = feature.pop("word_ids")

        # Create a map between words and corresponding token indices
        mapping = collections.defaultdict(list)
        current_word_index = -1
        current_word = None
        for idx, word_id in enumerate(word_ids):
            if word_id is not None:
                if word_id != current_word:
                    current_word = word_id
                    current_word_index += 1
                mapping[current_word_index].append(idx)

        # Randomly mask words
        mask = np.random.binomial(1, wwm_probability, (len(mapping),))
        input_ids = feature["input_ids"]
        labels = feature["labels"]
        new_labels = [-100] * len(labels)
        for word_id in np.where(mask)[0]:
            word_id = word_id.item()
            for idx in mapping[word_id]:
                new_labels[idx] = labels[idx]
                input_ids[idx] = tokenizer.mask_token_id
        feature["labels"] = new_labels

    return default_data_collator(features)

samples = [lm_datasets["train"][i] for i in range(2)] # 列表的形式
batch = whole_word_masking_data_collator(samples) # 去掉 key 为 "word_ids" 的键值，[mask]部分的值和"inout_ids"相同， labels全部负值为 -100


from transformers import TrainingArguments

batch_size = 64
# Show the training loss with every epoch
logging_steps = len(downsampled_dataset["train"]) // batch_size
# model_name = model_checkpoint.split("/")[-1]

output_dir='./output_dir'

training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    push_to_hub=True,
    fp16=True,
    logging_steps=logging_steps,
)

from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import default_data_collator
from tqdm.auto import tqdm
import torch
import math

batch_size = 64
train_dataloader = DataLoader(
    downsampled_dataset["train"],
    shuffle=True,
    batch_size=batch_size,
    collate_fn=data_collator,
)
eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, collate_fn=default_data_collator)

model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)

from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=5e-5)

from accelerate import Accelerator

accelerator = Accelerator()
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)

from transformers import get_scheduler

num_train_epochs = 10
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

progress_bar = tqdm(range(num_training_steps))

for epoch in range(num_train_epochs):
    # Training
    model.train()
    for batch in train_dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

    # Evaluation
    model.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(**batch)

        loss = outputs.loss
        losses.append(accelerator.gather(loss.repeat(batch_size)))

    losses = torch.cat(losses)
    losses = losses[: len(eval_dataset)]
    try:
        perplexity = math.exp(torch.mean(losses))
    except OverflowError:
        perplexity = float("inf")

    print(f">>> Epoch {epoch}: Perplexity: {perplexity}")

    # Save
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)

# Upload
# from huggingface_hub import get_full_repo_name
# repo_name = get_full_repo_name(model_name)
# print(repo_name) # disanda/distilbert-base-uncased-finetuned-imdb-accelerate
# repo = Repository(model_name, clone_from=repo_name)
# if accelerator.is_main_process:
#     tokenizer.save_pretrained(output_dir)
#     repo.push_to_hub(commit_message=f"Training in progress epoch {epoch}", blocking=False)

import torch
from transformers import pipeline
from transformers import AutoTokenizer
from transformers import AutoModelForMaskedLM

#model_pipe = pipeline("fill-mask", model="./output_dir")

model_checkpoint = "distilbert-base-uncased"
model = AutoModelForMaskedLM.from_pretrained('./output_dir')
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

text = "This is a great [MASK]."
inputs = tokenizer(text, return_tensors="pt")
token_logits = model(**inputs).logits
print(token_logits.shape)
print(inputs)
print(tokenizer.mask_token_id)

mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1] # Find the location of [MASK] and extract its logits
mask_token_logits = token_logits[0, mask_token_index, :]

top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist() # Pick the [MASK] candidates with the highest logits

for token in top_5_tokens:
    print(f"'>>> {text.replace(tokenizer.mask_token, tokenizer.decode([token]))}'")

# '>>> This is a great film.'
# '>>> This is a great movie.'
# '>>> This is a great idea.'
# '>>> This is a great one.'
# '>>> This is a great comedy.'






