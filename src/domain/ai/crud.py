from domain.ai.schema import *
from data_converter import return_to_dict
from chatbot import chat_with_llm

def deliver_to_model(text, scheduleId):
    final_text = [return_to_dict(text)]
    output = chat_with_llm(final_text, scheduleId)

    return {"model_output": output}