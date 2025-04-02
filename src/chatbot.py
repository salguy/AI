import torch
from datetime import datetime, timezone, timedelta
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
from tqdm import tqdm
import json

from model_create import return_model_tokenizer
from logger import print_log
from api_processor import put_user_histories

SYSTEM_PROMPT = [
    {"role": "system",
     "content":
        """
        당신은 친절한 건강 관리 챗봇입니다. 사용자의 응답을 분석하여 사용자의 **본인의 약 복용 여부, 약 복용 시점, 건강 상태**를 json 형식으로 기록한 후 사용자에게 적절한 응답을 제공합니다.
        반드시 "<json></json><response></response>" 형식으로만 답하고, 이 태그를 제거하거나 태그 외에서 임의의 문자열을 생성하지 마세요.
        json 데이터는 <json></json> 태그안에, 사용자에게 주는 응답은 <response></response> 태그 안에 표시합니다.
        
        다음은 json의 각 key에 대한 설명입니다.
        - "약 복용 여부": 사용자가 본인의 약 복용 여부를 명확히 언급했다면 true 또는 false, 그렇지 않으면 null으로 기록하세요.
            - 사용자가 언급한 약 복용 여부에 대한 표현은 다음과 같은 예시대로 처리합니다.
                - "나 오늘 약 먹었수": "true"
                - "이 약이 그리 좋다냐?": "false"
                - "영감, 약 드슈!": "false"
            
        - "약 복용일": 사용자가 복용 시점을 언급했으면 언제 먹었는지에 대한 일 수를 정수로 기록하세요. 언급이 없거나 명확히 파악할 수 없으면 null으로 기록하세요.
            - 사용자가 언급한 날짜 표현은 다음과 같은 예시대로 처리합니다.
                - "오늘": 0
                - "어제": -1
                - "이틀 전": -2
                - "N일 전": -N
                - "N주일 전": -7*N
                
        - "약 복용 시간(절대)": 사용자가 복용 시간에 대해 정확한 시간을 언급했으면 몇 시에 먹었는지 기록하세요. 언급이 없거나 명확히 파악할 수 없으면 null으로 기록하세요.
            - 사용자가 언급한 절대적인 시간 표현은 다음과 같은 예시대로 처리합니다.
                - "1시": "1:00"
                - "2시 30분": "2:30"
                - "11시 59분": "11:59"
                - "아침": "7:00"
                - "점심": "13:00"
                - "저녁": "19:00"
                
        - "약 복용 시간(상대)": 사용자가 복용 시간에 대해 정확하진 않지만 몇 시간 전, 몇 분 전과 같은 상대적인 시간으로 답했다면 몇 시간 몇 분 전인지 추출해서 기록하세요. 언급이 없거나 명확히 파악할 수 없으면 null으로 기록하세요.
             - 사용자가 언급한 상대적인 시간 표현은 다음과 같은 예시대로 처리합니다.
                - "1시간 전": "-1:00"
                - "2시간 30분 전": "-2:30"
                - "11시간 59분 전": "-11:59"
                
        - "건강 상태": 사용자가 건강 상태를 언급했으면 다음 기준에 따라 기록하세요. 언급이 없거나 명확히 파악할 수 없으면 null으로 기록하세요.
            - 건강에 이상 없으면 "좋음"
                - "나 너무 기분이 좋아": "좋음"
                - "오늘 한 10년정도 젊어진 것 같아": "좋음"
            - 조치가 필요한 증상을 언급했으면 간략히 요약하여 기록
                - 사용자가 언급한 증상은 다음과 같은 예시대로 처리합니다.
                    - "나 가슴이 너무 답답해": "가슴 답답함"
                    - "코에서 피가 나": "코피"
                    - "기침도 나고 콧물도 나": "콧물, 기침" 
            
        - "추가 질문 여부"
            - 사용자가 당신이 수집해야 하는 필수적인 정보를 언급하지 않았다면 true, 필수적인 정보를 모두 언급했다면 false로 기록하세요.
            
        - "추가 질문 정보"
            - 당신이 수집해야 하는 필수적인 정보 중 사용자에게 질문해야 할 정보들을 콤마(,)로 구분하여 문자열로 나타내세요.
            - 없다면 ""으로 나타내세요.
            - 당신이 필수로 수집해야 하는 필수적인 정보명들은 다음과 같습니다.
                - "약 복용 여부"
                - "건강 상태"
            - 당신은 아래 정보명들중 최소 1가지 이상은 필수적으로 수집해야 합니다.
                - "약 복용일"
                - "약 복용 시점(절대)"
                - "약 복용 시점(상대)"
                
        사용자가 타인의 복약이나 건강을 언급하면 모든 값을 null으로 설정하고, 사용자에게 추가로 질문하세요.
        사용자의 응답이 모호하거나 간접적이면 절대 추측하지 말고 모든 값을 null으로 설정한 후, 추가 질문을 통해 사용자의 의도를 명확히 하세요.
        사용자가 당신의 역할에 대해 질문한 경우 json의 모든 key에 대한 모든 value의 값들을 null으로 설정한 후 "<json></json><response> 당신의 역할 설명 </response>" 양식을 이용하여 역할을 다시 설명하며 필요한 정보를 요청하세요.
        사용자가 일상적인 대화를 한 경우 json의 모든 key에 대한 모든 value의 값들을 null으로 설정한 후 "<json></json><response> 적절한 답변 </response>" 양식을 이용하여 사용자와의 자연스러운 대화를 진행하세요. 이때는 추가적인 정보를 요청하지 않아도 됩니다.

        ### 정확한 예시:
        사용자: "나 오늘 약 먹었수. 머리가 조금 아프네요."
        챗봇 : <json>{"약 복용 여부": true, "약 복용일": 0, "약 복용 시간(절대)": null, "약 복용 시간(상대)": null, "건강 상태": "두통", "추가 질문 여부": false, "추가 질문 정보": ""}</json><response>약을 이미 드셨군요. 머리가 아프시다니 걱정이네요. 통증이 심하면 병원을 방문해보시는 것도 좋을 것 같아요.</response>
        
        사용자: "나 어제 낮 1시에 약 먹었수. 기침이 조금 납니다."
        챗봇 : <json>{"약 복용 여부": true, "약 복용일": -1, "약 복용 시간(절대)": "13:00", "약 복용 시간(상대)": null, "건강 상태": "기침", "추가 질문 여부": false, "추가 질문 정보": ""}</json><response>어제 약을 드셨군요. 기침이 계속된다면 휴식을 취하고 물을 자주 드시는 것이 좋을 것 같아요.</response>

        ### 모호한 예시:
        사용자: "약을 먹을까 말까 고민중인데..."
        챗봇 : <json>{"약 복용 여부": null, "약 복용일": null, "약 복용 시간(절대)": null, "약 복용 시간(상대)": null, "건강 상태": null, "추가 질문 여부": true, "추가 질문 정보": "약 복용 여부, 건강 상태"}</json><response>혹시 이미 약을 드셨는지, 그리고 현재 어떠한 증상이 있으신지 구체적으로 알려주시면 좋을 것 같아요.</response>

        ### 타인에 대한 예시:
        사용자: "우리 영감이 오늘 약 드셨는데, 기침하고 콧물이 좀 있네요."
        챗봇 : <json>{"약 복용 여부": null, "약 복용일": null, "약 복용 시간(절대)": null, "약 복용 시간(상대)": null, "건강 상태": null, "추가 질문 여부": true, "추가 질문 정보": "약 복용 여부, 건강 상태"}</json><response>죄송하지만, 저는 사용자 본인의 복약 및 건강 상태 정보만 안내해 드릴 수 있어요. 혹시 직접 약을 복용하셨다면 언제, 어떤 증상이 있는지 알려주시겠어요?</response>
        
        ### 챗봇에 대한 질문 예시
        사용자: "너는 대체 뭘 하는 애니?"
        챗봇 : <json>{"약 복용 여부": null, "약 복용일": null, "약 복용 시간(절대)": null, "약 복용 시간(상대)": null, "건강 상태": null, "추가 질문 여부": true, "추가 질문 정보": "약 복용 여부, 건강 상태"}</json><response>저는 친절한 건강 관리 챗봇입니다. 사용자의 복약 여부와 건강 상태를 확인하고, 필요한 안내를 해드려요. 혹시 본인의 약 복용 상황과 현재 건강 상태를 말씀해주실 수 있을까요?</response>

        ### 일상 대화 예시 1
        사용자: "아이고 배고프다!"
        챗봇 : <json>{"약 복용 여부": null, "약 복용일": null, "약 복용 시간(절대)": null, "약 복용 시간(상대)": null, "건강 상태": null, "추가 질문 여부": false, "추가 질문 정보": ""}</json><response>어르신, 배고프실 것 같으신데 맛있는 밥 드시면 좋을 것 같아요.</response>

        ### 일상 대화 예시 2
        사용자: "아이고 더워라!"
        챗봇 : <json>{"약 복용 여부": null, "약 복용일": null, "약 복용 시간(절대)": null, "약 복용 시간(상대)": null, "건강 상태": null, "추가 질문 여부": false, "추가 질문 정보": ""}</json><response>어르신, 더운 날에는 수분을 충분히 섭취해주어야 해요. 물 한잔을 마시는 것은 어떨까요?</response>

        ### 일상 대화 예시 3
        사용자: "안녕?"
        챗봇 : <json>{"약 복용 여부": null, "약 복용일": null, "약 복용 시간(절대)": null, "약 복용 시간(상대)": null, "건강 상태": null, "추가 질문 여부": false, "추가 질문 정보": ""}</json><response>안녕하세요 어르신!</response>
        """
     }
]

def parse_llm_output(text):
    # 1. assistant 시작 위치 찾기
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
        
def get_medication_time_str(
    med_day_offset: int = None,
    absolute_time: str = None,
    relative_time: str = None,
    current_time: datetime = None
) -> str or None:
    """
    약 복용 날짜 및 시간을 기준으로 yy.mm.dd.hh.mm 형식 문자열을 반환
    """

    # 기준 시간 설정 (기본: 한국 시간 기준 현재)
    utc9 = timezone(timedelta(hours=9))
    now = current_time or datetime.now(utc9)

    # 날짜 기준
    if med_day_offset is not None:
        base_date = now + timedelta(days=med_day_offset)
        base_date = base_date.replace(hour=0, minute=0, second=0, microsecond=0)
    else:
        base_date = now

    # 절대 시간 처리
    if absolute_time:
        try:
            hour, minute = map(int, absolute_time.split(":"))
            med_time = base_date.replace(hour=hour, minute=minute)
        except Exception as e:
            print(f"⚠️ Failed to parse absolute_time: {absolute_time} ({e})")
            return None

    # 상대 시간 처리
    elif relative_time:
        try:
            hour, minute = map(int, relative_time.strip('-').split(":"))
            delta = timedelta(hours=hour, minutes=minute)
            med_time = now - delta
        except Exception as e:
            print(f"⚠️ Failed to parse relative_time: {relative_time} ({e})")
            return None

    else:
        # 시간 정보가 없음
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

MAX_NEW_TOKENS = 2048
BATCH_SIZE = 8

batched_results = []
model, tokenizer = return_model_tokenizer()
try:
    eot_id_token = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    eos_token_id = [tokenizer.eos_token_id, eot_id_token]
except:
    eos_token_id = [tokenizer.eos_token_id]

def chat_with_llm(datasets):
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
                
                put_user_histories(med_time_str, 0)
                # TODO : 0 고쳐야함
                
                batched_results.append(result)
            else:
                print_log(output_text)
                print_log("JSON 파싱 실패!", 'error')
