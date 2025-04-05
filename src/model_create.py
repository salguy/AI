import torch
from datetime import datetime, timezone, timedelta
from transformers import AutoTokenizer, AutoModelForCausalLM

from logger import print_log

_model = None
_tokenizer = None

def return_model_tokenizer():
    global _model, _tokenizer

    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer

    def current_time_str_utc9():
        utc9 = timezone(timedelta(hours=9))
        return datetime.now(utc9).strftime('%y.%m.%d.%H.%M')

    NOW_TIME = current_time_str_utc9()
    TODAY_YEAR, TODAY_MONTH, TODAY_DATE, NOW_HOUR, NOW_MINUTE = NOW_TIME.split('.')

    model_name = "MLP-KTLim/llama-3-Korean-Bllossom-8B"
    
    print_log("Loading Model...")
    _tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    _tokenizer.pad_token_id = _tokenizer.eos_token_id

    _model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True
    )
    try:
        _model = torch.compile(_model)
    except RuntimeError:
        print_log("torch.compile() Failed. Check your PyTorch Version >= 2.0", 'error')

    print_log("Model Loaded.")
    return _model, _tokenizer
