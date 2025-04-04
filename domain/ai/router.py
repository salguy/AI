from fastapi import APIRouter, HTTPException

from domain.ai.crud import *
from domain.ai.schema import *

router = APIRouter()

@router.post("/api/inferences", summary="AI 추론")
async def ai_inference(record: AIInput):
    """
    AI의 추론을 반환하는 엔드포인트입니다.
    **input_text** : str
    """
    try:
        return deliver_to_model(record)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
