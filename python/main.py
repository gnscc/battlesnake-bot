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

    snake = game_state['you']['body']
    head = snake[0]

    possible_next_moves = {}
    for next_move in ['up', 'down', 'left', 'right']:
        
        possible_next_moves[next_move] = compute_next_coords(head, next_move)

    # Delete all movements that will collide with the snake
    possible_next_moves = {next_move: coords for next_move, coords in possible_next_moves.items() if not is_collision(snake, coords)}

    # Delete all movements that will hit the wall
    possible_next_moves = {next_move: coords for next_move, coords in possible_next_moves.items() if not is_wall(game_state, coords)}

    next_move = 'left'
    closest_food_at = 100000
    for possible_next_move in possible_next_moves:
        possible_next_move_coords = possible_next_moves[possible_next_move]

        for food in game_state['board']['food']:
            food_distance = abs(possible_next_move_coords['x'] - food['x']) + abs(possible_next_move_coords['y'] - food['y'])
            if food_distance < closest_food_at:
                next_move = possible_next_move
                closest_food_at = food_distance
            elif food_distance == closest_food_at:
                next_move = random.choice([next_move, possible_next_move]) # Randomize between the two']

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

# Start server when `python main.py` is run
if __name__ == '__main__':
    from server import run_server

    run_server({'info': info, 'start': start, 'move': move, 'end': end})
