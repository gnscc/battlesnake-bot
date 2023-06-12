import random
import typing as typ
from collections import deque

import numpy as np
import torch

from model import Linear_QNet, QTrainer

MAX_MEMORY = 100_000
BATCH_SIZE = 1000

class Agent:
    def __init__(self) -> None:
        self._n_games = 0
        self._last_state = None
        self._last_move = None
        self._last_snake_len = None
        self._memory = deque(maxlen=MAX_MEMORY) # popleft()
        self._model = Linear_QNet(11, 256, 3) # neural network
        self._trainer = QTrainer(self._model, lr=0.001, gamma=0.9)

    def info(self) -> typ.Dict:
        print('INFO')

        return {
            'apiversion': '1',
            'author': 'gnscc',  # Your Battlesnake Username
            'color': '#5e0a78', # Choose color
            'head': 'workout',  # Choose head
            'tail': 'weight',   # Choose tail
        }

    def start(self, game : typ.Dict) -> None:
        print('START')
        self._n_games += 1

    def move(self, game : typ.Dict) -> typ.Dict:
        print('MOVE')
        self._parse_game_coords_into_standard_coords(game)
        
        current_state = self._get_state(game)

        if self._last_move is not None and self._last_state is not None: # We should learn
            reward = self._get_reward(game)
            self._train_short_memory(self._last_state, self._last_move, reward, current_state, False)
            self._remember(self._last_state, self._last_move, reward, current_state, False)
        
        next_move = self._get_action(current_state)
        self._last_state = current_state
        self._last_move = next_move
        self._last_snake_len = len(game['you']['body'])

        return {'move': self._get_str_next_move(game, next_move)}
    
    def end(self, game : typ.Dict):
        print('END')
        self._parse_game_coords_into_standard_coords(game)
        current_state = self._get_state(game)

        if self._last_move is not None and self._last_state is not None: # We should learn
            reward = -10 #self._get_reward(current_state)
            self._train_short_memory(self._last_state, self._last_move, reward, current_state, True)
            self._remember(self._last_state, self._last_move, reward, current_state, True)

        self._train_long_memory()

    def _parse_game_coords_into_standard_coords(self, game : typ.Dict) -> None:
        '''
        This functions will parse the game dict given by the organization and convert it to the standard format.
        The coords give are based on the down left corner, whereas we want it to be at the top left corner.
        Therefore, we want to invert all the 'y's.

        This function will act over the same dict object so it won't return anything

        Parameters
        ----------
        game : dict
            The game dictionary.
        '''

        game_height = game['board']['height']
        for food in game['board']['food']:
            food['y'] = game_height - food['y'] - 1
        
        for segment in game['you']['body']:
            segment['y'] = game_height - segment['y'] - 1

        for rival_snake in game['board']['snakes']:
            for segment in rival_snake['body']:
                segment['y'] = game_height - segment['y'] - 1

    def _get_state(self, game : typ.Dict) -> np.ndarray:
        '''
        We will build a HxWxN, where H is the height of the board, W is the width of the board, and N is the number of features.
        We will use the following features:
        - Our snake body
        - Our snake head
        - Foods placed
        
        Parameters
        ----------
        game : dict
            The game dictionary.

        Returns
        -------
        np.ndarray
            The state.
        '''

        head = game['you']['body'][0]
        point_l = {'x': head['x'] - 1, 'y': head['y']}
        point_r = {'x': head['x'] + 1, 'y': head['y']}
        point_u = {'x': head['x'], 'y': head['y'] - 1}
        point_d = {'x': head['x'], 'y': head['y'] + 1}

        neck = game['you']['body'][1]
        dir_l = head['x'] < neck['x']
        dir_r = head['x'] > neck['x']
        dir_u = head['y'] < neck['y']
        dir_d = head['y'] > neck['y']

        food = game['board']['food'][0]

        state = [
            # Danger straight
            (dir_r and self._is_collision(game, point_r)) or
            (dir_l and self._is_collision(game, point_l)) or
            (dir_u and self._is_collision(game, point_u)) or
            (dir_d and self._is_collision(game, point_d)),

            # Danger right
            (dir_u and self._is_collision(game, point_r)) or
            (dir_d and self._is_collision(game, point_l)) or
            (dir_l and self._is_collision(game, point_u)) or
            (dir_r and self._is_collision(game, point_d)),

            # Danger left
            (dir_d and self._is_collision(game, point_r)) or
            (dir_u and self._is_collision(game, point_l)) or
            (dir_r and self._is_collision(game, point_u)) or
            (dir_l and self._is_collision(game, point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            food['x'] < head['x'],  # food left
            food['x'] > head['x'],  # food right
            food['y'] < head['y'],  # food u
            food['y'] > head['y']   # food d
        ]

        return np.array(state, dtype=int)
    
    def _is_collision(self, game: typ.Dict, point : typ.Dict) -> bool:
        '''
        Check if the point given is colliding with the wall.

        Parameters
        ----------
        game : dict
            The game dictionary.
        point : dict
            The point.

        Returns
        -------
        bool
            True if the point is colliding, false otherwise.
        '''

        # Board limits
        if game['board']['height'] <= point['y'] or game['board']['width'] <= point['x'] or point['y'] < 0 or point['x'] < 0:
            return True
        
        # Agent snake
        if point in game['you']['body']:
            return True

        # Other snakes body
        for snake in game['board']['snakes']:
            if point in snake['body']:
                return True

        return False
            
    def _get_reward(self, game : typ.Dict) -> int:
        '''
        We will give a reward of 10 if we eat a food, and -1 if we don't.

        Parameters
        ----------
        game : dict
            The game dictionary.
        current_state : np.ndarray
            The current state.

        Returns
        -------
        int
            The reward.
        '''
        
        if len(game['you']['body']) > self._last_snake_len:
            return 10
        
        return 0
    
    def _get_action(self, state : np.ndarray) -> np.ndarray:
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self._n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self._model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move
    
    def _get_str_next_move(self, game: typ.Dict, next_move : np.ndarray) -> str:
        '''
        Convert the next move into a string.

        Parameters
        ----------
        game : dict
            The game dictionary.
        
        next_move : np.ndarray
            The next move.

        Returns
        -------
        str
            The next move as a string.
        '''
        actions_str = ['right', 'down', 'left', 'up']

        head = game['you']['body'][0]
        neck = game['you']['body'][1]
        current_action_idx = 0
        if head['x'] > neck['x']: # right
            current_action_idx = 0
        if head['y'] > neck['y']: # down
            current_action_idx = 1
        if head['x'] < neck['x']: # left
            current_action_idx = 2
        if head['y'] < neck['y']: # up
            current_action_idx = 3

        if np.array_equal(next_move, [1, 0, 0]): # keep straight
            new_dir = actions_str[current_action_idx]
        elif np.array_equal(next_move, [0, 1, 0]): # right turn r -> d -> l -> u
            next_idx = (current_action_idx + 1) % 4
            new_dir = actions_str[next_idx]
        elif np.array_equal(next_move, [0, 0, 1]): # left turn r -> u -> l -> d
            next_idx = (current_action_idx - 1) % 4
            new_dir = actions_str[next_idx]

        return new_dir

    def _remember(self, state : np.ndarray, action : int, reward : int, next_state : np.ndarray, done : typ.Union[bool, np.ndarray]) -> None:
        '''
        Append states and rewards to memory to be able to retrain the model.

        Parameters
        ----------
        state : np.ndarray
            The state.
        action : int
            The action.
        reward : int
            The reward.
        next_state : np.ndarray
            The next state.
        '''

        self._memory.append((state, action, reward, next_state, done))

    def _train_short_memory(self, state : np.ndarray, action : np.ndarray, reward : typ.Union[int, np.ndarray],  next_state : np.ndarray, done : typ.Union[bool, np.ndarray]):
        self._trainer.train_step(state, action, reward, next_state, done)

    def _train_long_memory(self):
        if len(self._memory) <= 0:
            return
        elif len(self._memory) > BATCH_SIZE:
            mini_sample = random.sample(self._memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self._memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self._train_short_memory(states, actions, rewards, next_states, dones)

# Start server when `python agent.py` is run
if __name__ == '__main__':
    dqn_agent = Agent()


    from server import run_server

    run_server({'info': dqn_agent.info, 'start': dqn_agent.start, 'move': dqn_agent.move, 'end': dqn_agent.end})
