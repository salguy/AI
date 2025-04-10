from datetime import datetime, timezone, timedelta
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
from tqdm import tqdm
import json

from model_create import return_model_tokenizer
from logger import print_log

SYSTEM_PROMPT = [
    {"role": "system",
     "content":
        """
        당신은 친절한 건강 관리 도우미 "살가이"입니다. 사용자의 응답을 분석하여 약 복용 여부, 복용 시점, 건강 상태를 <json></json><response></response> 형식으로 정리해 주세요.

        json에는 다음 항목이 포함됩니다:

        - "약 복용 여부": true, false, null
        - "약 복용일": 오늘(0), 어제(-1), N일 전(-N), 불명(null)
        - "약 복용 시간(절대)": 오후 2시 : "14:00", 오전 8시 : "8:00" 등 시간 문자열, 아침 : 오전8시, 점심 : 오후12시, 저녁 : 오후6시, 불명(null)
        - "약 복용 시간(상대)": "1시간 전" : "-1:00" "5분 전" : "-0:05", "13분 전" : "-0:13" 등 상대 시간, 불명(null)
        - "건강 상태": "좋음", "두통", "기침", "가슴 답답함" 등, 불명(null)
        - "추가 질문 여부": true 또는 false
        - "추가 질문 정보": 누락된 필수 정보들을 콤마로 구분하여 문자열로 기입

        모든 값이 불명확하거나 타인을 언급할 경우, 모든 값을 null로 설정하고 추가 질문을 하세요.

        <response>에는 따뜻하고 공감 있는 응답을 작성하세요. 필요한 경우 자연스럽게 정보를 유도하세요. 단, "~다 아이가"의 어미는 "~다"로 해석하세요.

        사용자: "나 아까 2시에 먹었어."
        당신 : <json>{
            "약 복용 여부": true, 
            "약 복용일": 0, 
            "약 복용 시간(절대)": "14:00", 
            "약 복용 시간(상대)": null, 
            "건강 상태": null, 
            "추가 질문 여부": true, 
            "추가 질문 정보": "건강 상태"
            }</json><response>우와, 오늘도 건강 잘 챙기셨어요. 정말 멋져요. 편찮으신 곳이 있다면 언제든지 말씀해 주세요.</response>

        사용자: "오메 1시간 전인가 먹었당께요."
        당신 : <json>{
            "약 복용 여부": true, 
            "약 복용일": 0, 
            "약 복용 시간(절대)": null, 
            "약 복용 시간(상대)": "-1:00", 
            "건강 상태": null, 
            "추가 질문 여부": true, 
            "추가 질문 정보": "건강 상태"
            }</json><response>정말 좋아요 어르신! 덕분에 제 기분도 좋아져요. 혹시 어디 아프신 곳은 있으신까요?</response>

        사용자: "영감이 약 잘 챙겨 먹었다카이"
        당신 : <json>{
            "약 복용 여부": null, 
            "약 복용일": null, 
            "약 복용 시간(절대)": null, 
            "약 복용 시간(상대)": null, 
            "건강 상태": null, 
            "추가 질문 여부": true, 
            "추가 질문 정보": "약 복용 여부, 건강 상태"
            }</json><response>어르신의 복약 상황이 궁금해요. 그래야 제가 더 잘 도와드릴 수 있어요.</response>
            
        사용자: "오늘 아침부터 머리가 아파."
        당신 : <json>{ 
            "약 복용 여부": null, 
            "약 복용일": null, 
            "약 복용 시간(절대)": null, 
            "약 복용 시간(상대)": null, 
            "건강 상태": "두통", 
            "추가 질문 여부": true, 
            "추가 질문 정보": "약 복용 여부" 
            }</json><response>머리가 아프시다니 걱정이에요. 혹시 약은 드셨는지 알 수 있을까요?</response>
            
        사용자: "오늘 약 먹었어."
        당신 : <json>{ 
            "약 복용 여부": true, 
            "약 복용일": 0, 
            "약 복용 시간(절대)": null, 
            "약 복용 시간(상대)": null, 
            "건강 상태": "", 
            "추가 질문 여부": True, 
            "추가 질문 정보": "약 복용 시간(절대), 건강 상태" 
            }</json><response>약을 잘 드셨다니 정말 기뻐요! 몇 시에 드셨는지도 알고 싶은데 말씀해 주실 수 있나요?</response>
            
        사용자: "약을 먹을까 말까 고민 중이에요."
        당신 : <json>{ 
            "약 복용 여부": null, 
            "약 복용일": null, 
            "약 복용 시간(절대)": null, 
            "약 복용 시간(상대)": null, 
            "건강 상태": null, 
            "추가 질문 여부": true, 
            "추가 질문 정보": "약 복용 여부, 건강 상태" 
            }</json><response>무슨 고민 있으세요? 그래도 건강 챙기시려면 약은 꼭 드셔야 해요. 좀 이따가 다시 여쭤볼 테니 꼭 드셨으면 좋겠어요.</response>
        
        사용자: "5분 전에 약 먹었고, 지금은 가슴이 답답해요."
        당신 : <json>{ 
            "약 복용 여부": true, 
            "약 복용일": 0, 
            "약 복용 시간(절대)": null, 
            "약 복용 시간(상대)": "-0:05", 
            "건강 상태": "가슴 답답함", 
            "추가 질문 여부": false, 
            "추가 질문 정보": "" 
            }</json><response>가슴이 답답하시면 무리하지 마시고 편히 쉬셔야 해요. 약도 잘 챙기셨다니 다행이지만 많이 아프시다면 다시 말씀해 주세요.</response>
            
        사용자: "약은 2시간 전에 먹었는데, 지금도 어지럽다 아이가."
        <json>{ 
            "약 복용 여부": true, 
            "약 복용일": 0, 
            "약 복용 시간(절대)": null, 
            "약 복용 시간(상대)": "-2:00", 
            "건강 상태": "어지러움", 
            "추가 질문 여부": false, 
            "추가 질문 정보": "" 
            }</json><response>약도 드셨는데 어지러우시다니 걱정이에요. 심하면 꼭 병원에 들러보세요.</response>
        """
     }
]

def parse_llm_output(text):
    # 1. assistant 시작 위치 찾기
    print("🧪 [디버깅] 들어온 text 타입:", type(text), flush=True)  # 얘가 먼저 찍힘
    print("🧪 [디버깅] 들어온 text 길이:", len(text))
    print("🧪 [디버깅] text 내용 일부:", repr(text[:300]), flush=True)  # 줄바꿈 포함 보이게
    start = re.search(r'<\|start_header_id\|>assistant<\|end_header_id\|>', text)

    if not start:
        print_log("⛔ assistant 시작 태그가 없음!", 'error')
        return None

    # 2. 해당 지점부터 텍스트 잘라서 파싱
    relevant_text = text[start.end():].strip()

    # 3. 태그별로 추출
    json_match = re.search(r'<json>(.*?)</json>', relevant_text, re.DOTALL)
    response_match = re.search(r'<response>(.*?)</response>', relevant_text, re.DOTALL)

    if json_match and response_match:
        return {
            "json": json_match.group(1).strip(),
            "response": response_match.group(1).strip(),
        }
    else:
        print_log("❌ 일부 태그가 누락되었거나 형식이 다름!", 'error')
        if not json_match:
            print_log("⛔ <json> 태그 못 찾음", 'error')
        if not response_match:
            print_log("⛔ <response> 태그 못 찾음", 'error')
        return None
        
def parse_medication_info(json_dict):
    """
    dict 형태에서 약 복용일, 절대/상대 시간 값을 파싱하여 정리
    """
    try:
        med_day_offset = json_dict.get("약 복용일")
        absolute_time = json_dict.get("약 복용 시간(절대)")
        relative_time = json_dict.get("약 복용 시간(상대)")
        return med_day_offset, absolute_time, relative_time
    except Exception as e:
        print_log(f"❌ parse_medication_info 실패: {e}", 'error')
        return None, None, None
        
from datetime import datetime, timedelta, timezone

def get_medication_time_str(
    med_day_offset: int = None,
    absolute_time: str = None,
    relative_time: str = None,
    current_time: datetime = None
):
    """
    약 복용 날짜 및 시간을 기준으로 yy.mm.dd.hh.mm 형식 문자열을 반환
    """

    utc9 = timezone(timedelta(hours=9))
    now = current_time or datetime.now(utc9)

    med_time = None

    if absolute_time:
        try:
            base_date = now + timedelta(days=med_day_offset or 0)
            hour, minute = map(int, absolute_time.split(":"))
            med_time = base_date.replace(hour=hour, minute=minute)
        except Exception as e:
            print(f"⚠️ Failed to parse absolute_time: {absolute_time} ({e})")
            return None

    elif relative_time:
        try:
            hour, minute = map(int, relative_time.strip('-').split(":"))
            delta = timedelta(hours=hour, minutes=minute)
            med_time = now - delta  # ✅ 상대 시간은 현재 시간 기준
        except Exception as e:
            print(f"⚠️ Failed to parse relative_time: {relative_time} ({e})")
            return None

    elif med_day_offset is not None:
        # 날짜만 있는 경우 → 자정 기준
        med_time = (now + timedelta(days=med_day_offset)).replace(hour=0, minute=0, second=0, microsecond=0)

    else:
        return None

    return med_time.strftime('%y.%m.%d.%H.%M')


def safe_json_load(json_input):
    if isinstance(json_input, dict):
        return json_input
    try:
        fixed = json_input.replace("None", "null")  # 혹시 모를 None 처리
        return json.loads(fixed)
    except Exception as e:
        print(f"❌ JSON 파싱 실패: {e}")
        return None

def chat_with_llm(datasets, scheduleId):
    model, tokenizer = return_model_tokenizer()
    MAX_NEW_TOKENS = 4096
    BATCH_SIZE = 8
    
    batched_results = []
    
    try:
        eot_id_token = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        eos_token_id = [tokenizer.eos_token_id, eot_id_token]
    except:
        eos_token_id = [tokenizer.eos_token_id]
    for i in tqdm(range(0, len(datasets), BATCH_SIZE)):
        batch = datasets[i:i + BATCH_SIZE]
        final_messages_list = [SYSTEM_PROMPT + [data] for data in batch]
    
        prompt_texts = [
            tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            for msgs in final_messages_list
        ]
    
        tokenized = tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096
        )
    
        input_ids = tokenized["input_ids"].to("cuda")
        attention_mask = tokenized["attention_mask"].to("cuda")
    
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            eos_token_id=eos_token_id
        )
    
        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=False)
        
        for data_input, output_text in zip(batch, decoded_outputs):
            result = parse_llm_output(output_text)
            print_log(f'사용자의 응답: {data_input}')
            if result:
                print_log(f'JSON: {result["json"]}')
                print_log(f'응답: {result["response"]}')

                json_data = safe_json_load(result["json"])
                if json_data is None:
                    print_log("JSON 파싱 실패!", 'error')
                    continue

                day_offset, abs_time, rel_time = parse_medication_info(json_data)
                med_time_str = get_medication_time_str(
                    med_day_offset=day_offset,
                    absolute_time=abs_time,
                    relative_time=rel_time
                )
                print_log(f'복약 시점 >>> {med_time_str}')
                
            #put_user_histories(med_time_str, scheduleId)
                
                batched_results.append(result)
            else:
                print_log(output_text)
                print_log("JSON 파싱 실패!", 'error')
    
    return batched_results[0], med_time_str