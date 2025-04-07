from domain.ai.schema import *
from data_converter import return_to_dict
from chatbot import chat_with_llm
from logger import print_log


def deliver_to_model(text, scheduleId):
    print_log(f"💡 [AI모델] 받은 텍스트: {text}")
    final_text = [return_to_dict(text)]
    print_log(f"🧠 변환된 텍스트: {final_text}")

    output = chat_with_llm(final_text, scheduleId)
    print_log(f"🗣️ 모델 응답: {output}")

    return {"model_output": output}