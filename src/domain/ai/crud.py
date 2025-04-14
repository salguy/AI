from domain.ai.schema import AIInput
from prompts import *
from data_converter import return_to_dict
from chatbot import chat_with_llm
from logger import print_log




async def process_check_meal(text: str) -> str:
    """
    사용자의 식사 여부를 확인하고 적절한 응답을 생성합니다.

    Args:
        text (str): 사용자의 입력 텍스트

    Returns:
        str: AI 모델의 응답 텍스트

    Raises:
        ValueError: 처리 중 오류가 발생한 경우
    """
    print_log(f"💡 [AI모델] 받은 텍스트: {text}")
    try:
        # 입력 텍스트를 딕셔너리로 변환
        input_data = {"role": "user", "content": text}
        datasets = [input_data]
        
        # 식사 확인 프롬프트 사용
        response = chat_with_llm(datasets, custom_prompt=MEAL_CHECK_PROMPT)
        print_log(f"🗣️ 모델 응답: {response}")
        return response
    except Exception as e:
        print_log(f"❌ 식사 확인 처리 중 오류 발생: {str(e)}", "error")
        raise ValueError(f"식사 확인 처리 중 오류 발생: {str(e)}")

async def process_induce_medicine(text: str) -> str:
    """
    사용자에게 약 복용을 유도하는 응답을 생성합니다.

    Args:
        text (str): 사용자의 입력 텍스트

    Returns:
        str: AI 모델의 응답 텍스트

    Raises:
        ValueError: 처리 중 오류가 발생한 경우
    """
    print_log(f"💡 [AI모델] 받은 텍스트: {text}")
    try:
        input_data = {"role": "user", "content": text}
        datasets = [input_data]
        
        # 약 복용 유도 프롬프트 사용
        response = chat_with_llm(datasets, custom_prompt=MEDICINE_INDUCTION_PROMPT)
        print_log(f"🗣️ 모델 응답: {response}")
        return response
    except Exception as e:
        print_log(f"❌ 약 복용 유도 처리 중 오류 발생: {str(e)}", "error")
        raise ValueError(f"약 복용 유도 처리 중 오류 발생: {str(e)}")

async def process_notify_medicine() -> str:
    """
    사용자에게 약 복용 시간임을 알리는 응답을 생성합니다.

    Returns:
        str: AI 모델의 응답 텍스트

    Raises:
        ValueError: 처리 중 오류가 발생한 경우
    """
    print_log("💡 [AI모델] 약 복용 알림 요청")
    try:
        # 빈 대화 히스토리로 시작
        datasets = []
        response = chat_with_llm(datasets, custom_prompt=MEDICINE_NOTIFICATION_PROMPT)
        print_log(f"✅ [AI모델] 약 복용 알림 응답: {response}")
        return response
    except Exception as e:
        print_log(f"❌ 약 복용 알림 처리 중 오류 발생: {str(e)}", "error")
        raise ValueError(f"약 복용 알림 처리 중 오류 발생: {str(e)}")

async def process_confirm_medicine(text: str) -> str:
    """
    사용자의 약 복용 여부를 확인하고 적절한 응답을 생성합니다.

    Args:
        text (str): 사용자의 입력 텍스트

    Returns:
        str: AI 모델의 응답 텍스트

    Raises:
        ValueError: 처리 중 오류가 발생한 경우
    """
    print_log(f"💡 [AI모델] 받은 텍스트: {text}")
    try:
        input_data = {"role": "user", "content": text}
        datasets = [input_data]
        
        # 약 복용 확인 프롬프트 사용
        response = chat_with_llm(datasets, custom_prompt=MEDICINE_CONFIRMATION_PROMPT)
        print_log(f"🗣️ 모델 응답: {response}")
        return response
    except Exception as e:
        print_log(f"❌ 약 복용 확인 처리 중 오류 발생: {str(e)}", "error")
        raise ValueError(f"약 복용 확인 처리 중 오류 발생: {str(e)}")

async def deliver_to_model(input_text: str) -> dict:
    """
    AI 모델에 입력을 전달하고 응답을 반환합니다.

    Args:
        input_text (str): 사용자 입력 텍스트

    Returns:
        dict: AI 모델의 응답
    """
    print_log(f"💡 [AI모델] 받은 텍스트: {input_text}")
    try:
        # 입력 텍스트를 딕셔너리로 변환
        input_data = {"role": "user", "content": input_text}
        datasets = [input_data]
        
        # 모델에 입력 전달
        response = chat_with_llm(datasets)
        print_log(f"🗣️ 모델 응답: {response}")
        
        return {"model_output": response}
    except Exception as e:
        print_log(f"❌ 모델 응답 생성 중 오류: {str(e)}", "error")
        raise ValueError(f"모델 응답 생성 중 오류 발생: {str(e)}")
