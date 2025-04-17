from fastapi import APIRouter, HTTPException
from domain.ai.crud import *
from logger import print_log
from domain.ai.schema import *
from fastapi import Request


router = APIRouter()

@router.post("/api/inferences", summary="AI 추론")
async def ai_inference(record: AIInput, request: Request):
    """
    AI의 추론을 반환하는 엔드포인트입니다.
    **input_text** : str
    """
    try:
        raw = await request.body()
        raw = raw.decode("utf-8")
        print_log(f"🧾 [AI 서버] Raw body: {raw}")
        return deliver_to_model(record.input_text)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/api/inference/check_meal")
async def check_meal_inference(data: AIInput):
    """
    복약 전 식사여부 체크를 위한 AI 추론
    
    Parameters:
        - input_text: 사용자 음성을 텍스트로 변환한 내용
    
    Returns:
        - model_output: AI 모델의 출력
            - response: AI의 응답 텍스트
    """
    try:
        return await process_check_meal(data.input_text)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/inference/induce_medicine")
async def induce_medicine_inference(data: AIInput):
    """
    복약 유도를 위한 AI 추론
    
    Parameters:
        - input_text: 사용자 음성을 텍스트로 변환한 내용
    
    Returns:
        - model_output: AI 모델의 출력
            - response: AI의 응답 텍스트
    """
    try:
        return await process_induce_medicine(data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/inference/taking_medicine_time")
async def taking_medicine_time_inference():
    """
    복약 시점 도달을 위한 AI 추론
    
    Parameters:
        - input_text: 사용자 음성을 텍스트로 변환한 내용
    
    Returns:
        - model_output: AI 모델의 출력
            - response: AI의 응답 텍스트
    """
    try:
        return await process_notify_medicine()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/inference/check_medicine")
async def check_medicine_inference(data: AIInput):
    """
    복용 완료 확인을 위한 AI 추론
    
    Parameters:
        - input_text: 사용자 음성을 텍스트로 변환한 내용
    
    Returns:
        - model_output: AI 모델의 출력
            - response: AI의 응답 텍스트
    """
    try:
        return await process_confirm_medicine(data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/inference/daily_talk")
async def daily_talk_inference(data: AIInput):
    """
    일상 대화를 위한 AI 추론
    
    Parameters:
        - input_text: 사용자 음성을 텍스트로 변환한 내용
    
    Returns:
        - model_output: AI 모델의 출력
            - response: AI의 응답 텍스트
    """
    try:
        return await process_daily_talk(data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))