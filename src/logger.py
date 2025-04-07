from datetime import datetime
import colorama

def print_log(msg, type='info'):
    time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    red = colorama.Fore.RED
    yellow = colorama.Fore.YELLOW
    green = colorama.Fore.GREEN
    if(type == 'error'):
        print(red + f'[{time_str}] [ERROR] {msg}', flush=True)
    elif(type == 'warning'):
        print(yellow + f'[{time_str}] [WARNING] {msg}', flush=True)
    else:
        print(green + f'[{time_str}] [INFO] {msg}', flush=True)
    print(green, end = '', flush=True)