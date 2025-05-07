import gradio as gr
import requests

API_URL = "http://127.0.0.1:8000/api/inference"

def check_meal(text):
    response = requests.post(f"{API_URL}/check_meal", json={"input_text": text})
    return response.json().get("model_output", "오류 발생")

def induce_medicine(text):
    response = requests.post(f"{API_URL}/induce_medicine", json={"input_text": text})
    return response.json().get("model_output", "오류 발생")

def notify_medicine():
    response = requests.post(f"{API_URL}/taking_medicine_time")
    return response.json().get("model_output", "오류 발생")

def confirm_medicine(text):
    response = requests.post(f"{API_URL}/check_medicine", json={"input_text": text})
    return response.json().get("model_output", "오류 발생")

def daily_talk(text):
    response = requests.post(f"{API_URL}/daily_talk", json={"input_text": text})
    return response.json().get("model_output", "오류 발생")

def general_inference(text):
    response = requests.post(f"{API_URL}s", json={"input_text": text})
    return response.json().get("model_output", "오류 발생")

with gr.Blocks() as demo:
    gr.Markdown("# Salgai")
    
    with gr.Tab("식사 여부 확인"):
        meal_input = gr.Textbox(label="입력 텍스트")
        meal_output = gr.Textbox(label="응답")
        gr.Button("확인").click(fn=check_meal, inputs=meal_input, outputs=meal_output)

    with gr.Tab("약 복용 유도"):
        induce_input = gr.Textbox(label="입력 텍스트")
        induce_output = gr.Textbox(label="응답")
        gr.Button("유도").click(fn=induce_medicine, inputs=induce_input, outputs=induce_output)

    with gr.Tab("복약 알림"):
        notify_output = gr.Textbox(label="응답")
        gr.Button("알림").click(fn=notify_medicine, inputs=[], outputs=notify_output)

    with gr.Tab("복약 확인"):
        confirm_input = gr.Textbox(label="입력 텍스트")
        confirm_output = gr.Textbox(label="응답")
        gr.Button("확인").click(fn=confirm_medicine, inputs=confirm_input, outputs=confirm_output)

    with gr.Tab("일상 대화"):
        talk_input = gr.Textbox(label="입력 텍스트")
        talk_output = gr.Textbox(label="응답")
        gr.Button("대화").click(fn=daily_talk, inputs=talk_input, outputs=talk_output)

    with gr.Tab("일반 추론"):
        general_input = gr.Textbox(label="입력 텍스트")
        general_output = gr.Textbox(label="응답")
        gr.Button("추론").click(fn=general_inference, inputs=general_input, outputs=general_output)

demo.launch(share=True)