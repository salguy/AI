from fastapi import APIRouter, HTTPException
from domain.ai.crud import *
from logger import print_log



router = APIRouter()

@router.post("/api/inferences", summary="AI 추론")
async def ai_inference(record: AIInput):
    """
    AI의 추론을 반환하는 엔드포인트입니다.
    **input_text** : str
    **scheduleId**   : int
    """
    try:
        print_log("input_text : ", record.input_text)
        print_log("scheduleId : ", record.scheduleId)
        return deliver_to_model(record.input_text, record.scheduleId)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))