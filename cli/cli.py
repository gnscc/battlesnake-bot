import os
import subprocess
import time

import config as cfg

#  ./battlesnake play -W 11 -H 11 --name 'Python Starter Project' --url http://localhost:8000 -g solo --browser

bin_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'battlesnake')
command = [bin_path, 'play', '-W', str(cfg.W), '-H', str(cfg.H)]

for snake in cfg.SNAKES:
    command += ['--name', snake[0], '--url', snake[1]]

command += ['-g', 'solo' if len(cfg.SNAKES) == 1 else 'standard']

if cfg.BROWSER:
    command += ['--browser']

for i in range(cfg.N_GAMES):
    time.sleep(cfg.TIME_BETWEEN_GAMES)

    subprocess.call(command)
    print("Game {} done.".format(i + 1))