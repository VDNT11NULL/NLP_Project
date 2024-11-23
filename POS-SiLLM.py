import os
import sys
import pdb
import fire
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, AutoTokenizer
from datasets import load_dataset
from utils.callbacks import Iteratorize, Stream
from utils.prompter import Prompter
import json
import csv
import stanza
import numpy as np

stanza.download('hi')

nlp = stanza.Pipeline(lang='hi', processors='tokenize,pos')

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass


def main(
    load_8bit: bool = False,
    base_model: str = "",
    lora_weights: str = "tloen/alpaca-lora-7b",
    prompt_template: str = "",  # The prompt template to use, will default to alpaca.
    data_path: str = "",
    output_translation_path: str = "",
    # waitk: int = 1,
    csv_output_path: str = "output_POS_bhavik_verb.csv",  # CSV file to store real-time updates
):
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    prompter = Prompter(prompt_template)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
        )
    elif device == "mps":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )

    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    def evaluate(
        instruction,
        input=None,
        output=None,
        suppress_tokens=None,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=128,
        stream_output=False,
        **kwargs,
    ):
        prompt = prompter.generate_prompt(instruction, input, output)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            num_beams=num_beams,
            suppress_tokens=suppress_tokens,
            **kwargs,
        )

        # Without streaming
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s, skip_special_tokens=True)
        return prompter.get_response(output), s.size(-1) - input_ids.size(-1)

    def pass_string(input):
        lst = input
        doc = nlp(lst)
        words = doc.sentences[-1].words
        n= len(words)
        i= 0
        indexes = []
        for i in range(n):
            if words[i].upos == 'VERB': #or words[i].upos == 'NOUN':
                indexes.append(i)
            # elif words[i].upos == 'ADJ' and np.random.random() < 0.7:
            #     indexes.append(i)
            # elif words[i].upos == 'ADV' and np.random.random() < 0.5:
            #     indexes.append(i)
            # elif words[i].upos == 'ADP' and np.random.random() < 0.3:
            #     indexes.append(i)
            # elif words[i].upos == 'DET' and np.random.random() < 0.1:
            #     indexes.append(i)
            # elif words[i].upos == 'INTJ' and np.random.random() < 0.2:
            #     indexes.append(i)
            # elif words[i].upos == 'CCONJ' and np.random.random() < 0.2:
            #     indexes.append(i)
                
        if len(indexes)==0 or indexes[-1] != n-1:
            indexes.append(n-1)
        return indexes

    def POS_policy(
        instruction,
        input=None,
        num_beams=1,
        # waitk=1,
        max_new_tokens=256
    ):
        cur_target_str = ""
        tokenized_input = input
        src_len = len(input.split())
        tmp_max_new_tokens = 1
        rw_seq = []
        # first_time = True
        suppress_tokens=[2]
        indexes = pass_string(input)
        
        for i in indexes:
            cut_input = ' '.join(input.split()[:i])
            tmp_max_new_tokens = 5
            if i == indexes[-1]:
                tmp_max_new_tokens = max_new_tokens
                suppress_tokens = None
            cur_target_str, tmp_size = evaluate(instruction, cut_input, output=cur_target_str, suppress_tokens=suppress_tokens, num_beams=num_beams, max_new_tokens=tmp_max_new_tokens)
            if i != indexes[-1]:
                cur_target_str = ' '.join(cur_target_str.split()[:i + 1])
                rw_seq.append(i)
                if cur_target_str.find('</s>') != -1:
                    break
            else:
                tmp_size = len(cur_target_str.split()) - i
                rw_seq = rw_seq + [src_len] * tmp_size
        rw_seq.append(src_len)

        return rw_seq, cur_target_str
    
    # Open the CSV file in append mode
    with open(csv_output_path, mode='a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['instruction', 'input', 'rw', 'translation']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write headers if file is empty (i.e. first time writing)
        if csvfile.tell() == 0:
            writer.writeheader()

        data = load_dataset("json", data_files=data_path)
        test_data = data["train"]
        output_text = []
        j = 1
        for item_data in test_data:
            print(item_data)
            print('sample' + str(j))
            j += 1
            tmp_result = POS_policy(item_data["instruction"], item_data["input"], num_beams=1, max_new_tokens=52)
            print('tmp result', tmp_result)
            index = tmp_result[1].find('\n')
            tmp_str = tmp_result[1]
            if index != -1:
                tmp_str = tmp_result[1][:index]
            output_text.append({'rw': tmp_result[0], 'translation': tmp_str})

            # Log the output to CSV
            writer.writerow({
                'instruction': item_data["instruction"],
                'input': item_data["input"],
                'rw': tmp_result[0],
                'translation': tmp_str
            })
            print(f"Logged to CSV: {item_data['instruction']} -> {tmp_str}")

        # Save final output to JSON
        with open(output_translation_path, "w", encoding='utf-8') as fp:
            json.dump(output_text, fp, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    fire.Fire(main)
