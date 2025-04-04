from inference import model_inference
from domain.ai.schema import *

def deliver_to_model(text):
    output = model_inference(text)


    return {"model_output": output}