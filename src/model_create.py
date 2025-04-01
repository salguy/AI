import torch
from datetime import datetime, timezone, timedelta
from transformers import AutoTokenizer, AutoModelForCausalLM
import re

from logger import print_log

def return_model_tokenizer():
    def current_time_str_utc9():
        utc9 = timezone(timedelta(hours=9))
        return datetime.now(utc9).strftime('%y.%m.%d.%H.%M')
    
    NOW_TIME = current_time_str_utc9()
    TODAY_YEAR, TODAY_MONTH, TODAY_DATE, NOW_HOUR, NOW_MINUTE = NOW_TIME.split('.')
    
    model_name = "MLP-KTLim/llama-3-Korean-Bllossom-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map={"default": 0},
        trust_remote_code=True
    )
    #model = model.to('cuda')
    try:
        model = torch.compile(model)
    except RuntimeError:
        print_log("torch.compile() 실패: PyTorch 2.0 이상이 아니거나 해당 모델이 호환되지 않을 수 있습니다.", 'error')
    return model, tokenizer