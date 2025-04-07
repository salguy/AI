import requests
import json

from logger import print_log

from dotenv import load_dotenv
import os

dotenv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../env/.env"))
load_dotenv(dotenv_path)

API_SERVER_URL = os.getenv("API_SERVER_URL") 
def put_user_histories(taken_at, schedule_id):
    data = {
        'taken_at' : taken_at,
        'schedule_id' : schedule_id
    }

    headers = {
        'accept' : 'application/json',
        'Content-Type': 'application/json'
    }

    url = f"{API_SERVER_URL}api/user/histories"

    response = requests.put(url, headers=headers, data=json.dumps(data))

    if response.status_code == 422:
        print_log(f'Put Status Code : {response.status_code}', 'error')
        print_log(f'Put Response Body : {response.text}', 'error')
    elif response.status_code == 200:
        print_log(f'Put Status Code : {response.status_code}')
        print_log(f'Put Response Body : {response.text}')