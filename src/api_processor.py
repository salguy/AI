import requests
import json

from logger import print_log

def put_user_histories(taken_at, schedule_id):
    data = {
        'taken_at' : taken_at,
        'schedule_id' : schedule_id
    }

    headers = {
        'accept' : 'application/json',
        'Content-Type': 'application/json'
    }

    url = "http://3.34.179.85:8000/api/user/histories"

    response = requests.put(url, headers=headers, data=json.dumps(data))

    if response.status_code == 422:
        print_log(f'Status Code : {response.status_code}', 'error')
        print_log(f'Response Body : {response.text}', 'error')
    elif response.status_code == 200:
        print_log(f'Status Code : {response.status_code}')
        print_log(f'Response Body : {response.text}')