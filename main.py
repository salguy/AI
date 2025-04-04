from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re


def model_inference(input_text: str):
    model = AutoModelForCausalLM.from_pretrained(
        "NCSOFT/Llama-VARCO-8B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("NCSOFT/Llama-VARCO-8B-Instruct")
    TODAY_DATE = datetime.date.today()
    messages = [
        {"role": "system",
        "content":
            f'''당신은 친절한 건강 관리 챗봇입니다. 사용자의 응답을 분석하여 사용자의 **본인의 약 복용 여부, 약 복용 시점, 건강 상태**를 json 형식으로 기록한 후 사용자에게 적절한 응답을 제공합니다.
        
            반드시 "<json></json><response></response>" 형식으로만 답하고, 이 태그 외에서 임의의 문자열을 생성하지 말아주세요.
        
            json 데이터는 <json></json> 태그안에, 사용자에게 주는 응답은 <response></response> 태그 안에 표시합니다.
        
            다음은 json의 각 key에 대한 설명입니다.
        
            - "약 복용 여부": 사용자가 본인의 약 복용 여부를 명확히 언급했다면 true 또는 false, 그렇지 않으면 None
            - "약 복용 시점": 본인의 약 복용을 언급했으면 시점 ("오늘" 혹은 "n일전")으로 기록, 그렇지 않으면 None
            - "건강 상태": 본인의 건강 상태를 언급했으면 다음 기준에 따라 기록하세요.
            - 건강에 별다른 이상 없으면 "좋음"
            - 전문가의 조치가 필요한 증상을 명확히 언급했다면 그 증상을 간략히 요약하여 기록
            - 건강 상태를 언급하지 않았다면 None
        
            사용자가 본인이 아닌 타인의 복약이나 건강 상태에 대해 이야기한 경우는 사용자의 정보로 간주하지 말고 모든 값을 None으로 설정하세요.
        
            사용자의 응답이 모호하거나 간접적 표현일 경우에는 절대 추측하지 말고, 모든 값을 None으로 설정한 후 사용자의 의도를 명확히 확인하도록 추가 질문을 제안하세요.
        
            예시:
            <json>{{"약 복용 여부": true, "약 복용 시점": "오늘", "건강 상태": "좋음"}}</json>
            <response>약을 잘 챙겨 드셨군요! 오늘 하루도 건강하게 보내세요.</response>
        
            모호한 표현 예시:
            응답:
            <json>{{"약 복용 여부": None, "약 복용 시점": None, "건강 상태": None}}</json>
            <response>메시지에서 복약 여부와 건강 상태를 정확히 파악하기 어렵습니다. 약을 언제 드셨는지 얘기해주시겠어요? 그리고 건강은 어떠세요?</response>'''}
    ]



    eos_token_id = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    data = {"role": "user", "content": input_text}
    final_message = messages + [data]
    inputs = tokenizer.apply_chat_template(final_message, return_tensors="pt").to(model.device)

    outputs = model.generate(
        inputs,
        eos_token_id=eos_token_id,
        max_length=8192,
        temperature=0.2,
        do_sample=False
    )

    output_text = tokenizer.decode(outputs[0])
    # assistant의 응답 부분만 정확히 추출
    match = re.search(
        r'<\|start_header_id\|>assistant<\|end_header_id\|>\s*(<json>.*?</json>.*?<response>.*?</response>)',
        output_text, re.DOTALL)
    if match:
        print(match.group(1).strip())
    else:
        print("적절한 응답을 찾지 못했습니다.")
            
            

datasets = [
    {"role": "user", "content": "나 어제 알약 먹었어 잘 했지?"},
    {"role": "user", "content": "나 가슴이 답답해, 약은 어제 먹었어."},
    {"role": "user", "content": "나 오늘 기분이 좋고, 약도 오늘 챙겨 먹었어."},
    {"role": "user", "content": "영감! 약 드슈"},
    {"role": "user", "content": "우리 영감이 어제 약도 잘 묵고 한 10년은 젊어진 느낌이 난다니깐?"},
    {"role": "user", "content": "약을 어떻게 먹어야 할까?"},
    {"role": "user", "content": "이 로봇이 말을 하네?"},
    {"role": "user", "content": "오늘 아침에 사과를 먹었수."},
    {"role": "user", "content": "뭐라고?"},
]