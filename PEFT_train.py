from datasets import load_dataset

ds = load_dataset("cais/mmlu", "auxiliary_train", streaming=True)
ds.num_shards = 2
# len(ds['train'])

from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import torch
os.environ['HF_TOKEN'] = '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'

tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-3B')
model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.2-3B')
tokenizer.pad_token = tokenizer.eos_token
# model.to(torch.bfloat16)

from peft import LoraConfig, get_peft_model
peft_config = LoraConfig(
    r=16,                        # Rank of the decomposition
    lora_alpha=32,               # Scaling factor
    lora_dropout=0.1,            # Dropout rate
    bias="none",                 # Options: "none", "all", "lora_only"
    task_type="CAUSAL_LM"  ,      # Task type (e.g., "CAUSAL_LM" for language modeling)
    target_modules=["q_proj", "k_proj", 'v_proj', 'o_proj']
)
# Wrap the model with PEFT
model = get_peft_model(model, peft_config)

# Print trainable parameters for verification
model.print_trainable_parameters()

base_prompt = 'Answer the following MCQ Question\n\n {question} based on Options given below\n\n Option1: {option1}\n Option2: {option2}\n Option3: {option3}\n Option4: {option4}\n\n  Answer: {answer} <|end_of_text|>'
def collate_fn(items):
    texts = []
    for item in items:
        itemm = item['train']
        texts.append(base_prompt.format(question = itemm['question'], option1 = itemm['choices'][0], option2 = itemm['choices'][1], option3 = itemm['choices'][2], option4 = itemm['choices'][3], answer = str(itemm['answer'])))
    tokenized = tokenizer(texts, return_tensors='pt', padding=True)
    tokenized.update({'labels':tokenized['input_ids']})
    return tokenized

from transformers import Trainer, TrainingArguments
import torch
# model = torch.compile(model, backend=)
training_args = TrainingArguments(output_dir='./results/', do_train=True, per_device_train_batch_size=2, num_train_epochs=3, remove_unused_columns=False ,learning_rate=2e-5, torch_compile=False, save_strategy='steps', report_to="none", lr_scheduler_type='cosine', warmup_ratio = 0.06, logging_steps=10000, run_name = 'NLP_peft', bf16=True, max_steps=99842, dataloader_num_workers=2, resume_from_checkpoint='results/checkpoint-1000')
trainer = Trainer(model=model, args=training_args, train_dataset=ds['train'], data_collator=collate_fn)

trainer.train()