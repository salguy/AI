import requests
import json

from logger import print_log

from dotenv import load_dotenv
import os

dotenv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../env/.env"))
load_dotenv(dotenv_path)

API_SERVER_URL = os.getenv("API_SERVER_URL") 
def put_user_histories(taken_at, schedule_id):
    url = f"{API_SERVER_URL}api/user/histories"
    payload = {
        "taken_at": taken_at,
        "schedule_id": schedule_id
    }
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json"
    }

    try:
        print_log(f"📤 PUT 요청: {payload} → {url}")
        response = requests.put(url, headers=headers, json=payload, timeout=5)

        print_log(f"📥 응답 코드: {response.status_code}")
        print_log(f"📦 응답 내용: {response.text}")

        if response.status_code == 422:
            print_log(f"❌ 요청 형식 오류 (422)", "error")
        elif not response.ok:
            print_log(f"❌ 기타 오류: {response.status_code}", "error")

    except requests.exceptions.RequestException as e:
        print_log(f"❌ PUT 요청 중 예외 발생: {e}", "error")