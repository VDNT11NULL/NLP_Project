k=7
Base_Model=meta-llama/Llama-3.2-1B-Instruct
LoRA_Weithts=weights/
Output_Translation=HMT_Policy/hin_eng_Wait_K.json
Test_Data=SFT_data/hmt_data_valid.json

python Wait-k-SiLLM.py \
    --base_model ${Base_Model} \
    --lora_weights ${LoRA_Weithts} \
    --prompt_template 'Text_translation' \
    --data_path ${Test_Data} \
    --output_translation_path ${Output_Translation} \
    --waitk ${k}
