import os
from art import tprint
from colorama import init

from logger import print_log

def environment_setting():
    requirements_file = os.path.join(os.path.dirname(__file__), '..', 'env', 'requirements.txt')
    os.system(f'pip install --quiet -r {requirements_file}')

def boot_system():
    tprint('Salguy')
    version_info_file = os.path.join(os.path.dirname(__file__), '..', 'env', 'version_info.txt')
    text = ''
    with open(version_info_file, 'r') as f:
        text = f.readlines()
    program_version = text[0].split('=')[0].strip()
    AI_version = text[1].split('=')[0].strip()
    print_log(f'PROGRAM VERSION \t\t | {program_version}')
    print_log(f'AI VERSION \t\t | {AI_version}')

if __name__ == '__main__':
    boot_system()
    environment_setting()