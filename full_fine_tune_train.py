from datasets import load_dataset
import torch

ds = load_dataset("cais/mmlu", "auxiliary_train", streaming=True)

from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-3B')
model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.2-3B')
tokenizer.pad_token = tokenizer.eos_token
model.to(torch.bfloat16)
base_prompt = 'Answer the following MCQ Question\n\n {question} based on Options given below\n\n Option1: {option1}\n Option2: {option2}\n Option3: {option3}\n Option4: {option4}\n\n  Answer: {answer} <|end_of_text|>'
def collate_fn(items):
    texts = []
    for item in items:
        item = item['train']
        texts.append(base_prompt.format(question = item['question'], option1 = item['choices'][0], option2 = item['choices'][1], option3 = item['choices'][2], option4 = item['choices'][3], answer = str(item['answer'])))
    tokenized = tokenizer(texts, return_tensors='pt', padding=True)
    tokenized.update({'labels':tokenized['input_ids']})
    return tokenized


from transformers import Trainer, TrainingArguments
training_args = TrainingArguments(output_dir='./results/', do_train=True, per_device_train_batch_size=4, num_train_epochs=3, remove_unused_columns=False ,learning_rate=2e-5, torch_compile=True, save_strategy='no', lr_scheduler_type='cosine', warmup_ratio = 0.06, logging_steps=20,report_to="wandb", run_name = 'NLP', dataloader_num_workers=4, max_steps=99842)
trainer = Trainer(model=model, args=training_args, train_dataset=ds['train'], data_collator=collate_fn)

trainer.train()

torch.save(model.cpu(), 'nlp_model_3B.pt')