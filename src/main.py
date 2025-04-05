import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from colorama import init

from data_converter import return_to_dict
from chatbot import chat_with_llm
from logger import print_log
from endpoint_runner import endpoint_run

def environment_setting():
    requirements_file = os.path.join(os.path.dirname(__file__), '..', 'env', 'requirements.txt')
    os.system(f'pip install --quiet -r {requirements_file}')
    
def boot_system():
    init()
    version_info_file = os.path.join(os.path.dirname(__file__), '..', 'env', 'version_info.txt')
    text = ''
    with open(version_info_file, 'r') as f:
        text = f.readlines()
    try:
        program_version = text[0].split('=')[1].strip()
        AI_version = text[1].split('=')[1].strip()
    except IndexError:
        print_log('Error: version_info.txt is not in the expected format.')
        program_version = 'unknown'
        AI_version = 'unknown'
    print_log(f'PROGRAM VERSION \t\t | {program_version}')
    print_log(f'AI VERSION \t\t | {AI_version}')
    endpoint_run()

if __name__ == '__main__':
    environment_setting()
    boot_system()
            