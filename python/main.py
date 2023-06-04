# Welcome to
# __________         __    __  .__                               __
# \______   \_____ _/  |__/  |_|  |   ____   ______ ____ _____  |  | __ ____
#  |    |  _/\__  \\   __\   __\  | _/ __ \ /  ___//    \\__  \ |  |/ // __ \
#  |    |   \ / __ \|  |  |  | |  |_\  ___/ \___ \|   |  \/ __ \|    <\  ___/
#  |________/(______/__|  |__| |____/\_____>______>___|__(______/__|__\\_____>
#
# This file can be a nice home for your Battlesnake logic and helper functions.
#
# To get you started we've included code to prevent your Battlesnake from moving backwards.
# For more info see docs.battlesnake.com

import random
import typing

from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder

# info is called when you create your Battlesnake on play.battlesnake.com
# and controls your Battlesnake's appearance
# TIP: If you open your Battlesnake URL in a browser you should see this data
def info() -> typing.Dict:
    print('INFO')

    return {
        'apiversion': '1',
        'author': 'gnscc',  # Your Battlesnake Username
        'color': '#5e0a78', # Choose color
        'head': 'workout',  # Choose head
        'tail': 'weight',   # Choose tail
    }


# start is called when your Battlesnake begins a game
def start(game_state: typing.Dict):
    print('GAME START')


# end is called when your Battlesnake finishes a game
def end(game_state: typing.Dict):
    print('GAME OVER\n')


# move is called on every turn and returns your next move
# Valid moves are 'up', 'down', 'left', or 'right'
# See https://docs.battlesnake.com/api/example-move for available data
def move(game_state: typing.Dict) -> typing.Dict:
    game_state_to_standard_coords(game_state)

    matrix_row = [1 for _ in range(game_state['board']['width'])]
    matrix_map = [matrix_row.copy() for _ in range(game_state['board']['height'])]

    snake = game_state['you']['body']
    for segment in snake:
        matrix_map[segment['y']][segment['x']] = 0
    
    food = game_state['board']['food']

    for row in matrix_map:
        print(row)

    grid = Grid(matrix=matrix_map)
    start = grid.node(snake[0]['x'], snake[0]['y'])
    finder = AStarFinder(diagonal_movement=DiagonalMovement.never)

    closest_path = None
    closest_distance = None
    for food in game_state['board']['food']:
        end = grid.node(food['x'], food['y'])

        path, runs = finder.find_path(start, end, grid)

        print(snake[0], food, path)

        if not len(path):
            continue

        if closest_distance is None or len(path) < closest_distance:
            closest_distance = len(path)
            closest_path = path

    next_point = snake[0]
    if closest_path is not None:
        next_point = {'x': closest_path[1][0], 'y': closest_path[1][1]}

    next_move = get_next_move(snake[0], next_point)
    print(f'MOVE {game_state["turn"]}: {next_move}')
    return {'move': next_move}


def compute_next_coords(current_coords: typing.Dict, move: str) -> typing.Dict:
    x = current_coords['x']
    y = current_coords['y']

    if move == 'up':
        y += 1
    elif move == 'down':
        y -= 1
    elif move == 'left':
        x -= 1
    elif move == 'right':
        x += 1

    return {'x': x, 'y': y}

def is_collision(snake: typing.List, coords: typing.Dict) -> bool:
    for segment in snake:
        if segment['x'] == coords['x'] and segment['y'] == coords['y']:
            return True
    return False

def is_wall(game_state: typing.Dict, coords: typing.Dict) -> bool:
    board_width = game_state['board']['width']
    board_height = game_state['board']['height']

    if coords['x'] < 0 or coords['x'] >= board_width:
        return True
    if coords['y'] < 0 or coords['y'] >= board_height:
        return True

    return False

def get_next_move(origin_point : typing.Dict, target_point : typing.Dict) -> str:
    origin_x = origin_point['x']
    origin_y = origin_point['y']

    target_x = target_point['x']
    target_y = target_point['y']

    if origin_x > target_x:
        return 'left'
    elif origin_x < target_x:
        return 'right'
    elif origin_y > target_y:
        return 'up'
    else:
        return 'down'
    
def game_state_to_standard_coords(game_state: typing.Dict) -> typing.List:
    game_height = game_state['board']['height']
    for food in game_state['board']['food']:
        food['y'] = game_height - food['y'] - 1
    
    for segment in game_state['you']['body']:
        segment['y'] = game_height - segment['y'] - 1

# Start server when `python main.py` is run
if __name__ == '__main__':
    from server import run_server

    run_server({'info': info, 'start': start, 'move': move, 'end': end})
