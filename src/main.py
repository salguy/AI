import os
from colorama import init

from data_converter import return_to_dict
from chatbot import chat_with_llm
from logger import print_log

def environment_setting():
    requirements_file = os.path.join(os.path.dirname(__file__), '..', 'env', 'requirements.txt')
    os.system(f'pip install --quiet -r {requirements_file}')

def boot_system():
    init()
    version_info_file = os.path.join(os.path.dirname(__file__), '..', 'env', 'version_info.txt')
    text = ''
    with open(version_info_file, 'r') as f:
        text = f.readlines()
    program_version = text[0].split('=')[1].strip()
    AI_version = text[1].split('=')[1].strip()
    print_log(f'PROGRAM VERSION \t\t | {program_version}')
    print_log(f'AI VERSION \t\t | {AI_version}')

if __name__ == '__main__':
    environment_setting()
    boot_system()
    while True:
        text = input('메세지를 입력하세요, -1을 입력하면 종료시킵니다. | ')
        if(text.strip()=='-1'):
            break
        else:
            final_text = [return_to_dict(text)]
            chat_with_llm(final_text)
            