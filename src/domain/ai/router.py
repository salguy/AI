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
    **scheduleId**   : int
    """
    try:
        raw = await request.body()
        print_log(f"🧾 [AI 서버] request :{request}")
        print_log(f"🧾 [AI 서버] Raw body: {raw.decode("utf-8")}")
        print_log(f"response : {record}")
        print_log(f"input_text : {record.input_text}")
        print_log(f"scheduleId : {record.scheduleId}")
        return deliver_to_model(record.input_text, record.scheduleId)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))