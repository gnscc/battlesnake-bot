import typing as typ
from collections import deque
import numpy as np
import random
from model import CNN_QNet, QTrainer

MAX_MEMORY = 100_000
BATCH_SIZE = 1000

class Agent:
    def __init__(self) -> None:
        self._n_games = 0
        self._last_state = None
        self._last_move = None
        self._memory = deque(maxlen=MAX_MEMORY) # popleft()
        self._model = CNN_QNet([11, 11, 3])
        self._trainer = QTrainer(self._model, lr=0.001, gamma=0.9)

    def info() -> typ.Dict:
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
            reward = self._get_reward(current_state)
            self._train_short_memory(self._last_state, self._last_move, reward, current_state)
            self._remember(self._last_state, self._last_move, reward, current_state)
        
        next_move = self._get_action()
        self._last_state = current_state
        self._last_move = next_move

        return {'move': next_move}
    
    def end(self, game : typ.Dict):
        print('END')
        self._parse_game_coords_into_standard_coords(game)
        current_state = self._get_state(game)

        if self._last_move is not None and self._last_state is not None: # We should learn
            reward = -10 #self._get_reward(current_state)
            self._train_short_memory(self._last_state, self._last_move, reward, current_state)
            self._remember(self._last_state, self._last_move, reward, current_state)

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

        state = np.zeros((game['board']['height'], game['board']['width'], 3)) 
        
        # First channel will be our snake body
        our_snake = state['you']['body']

        for segment in our_snake:
            state[segment['y'], segment['x'], 0] = 1

        # Second channel will be our snake head
        our_snake_head = state['you']['body'][0]
        state[our_snake_head['y'], our_snake_head['x'], 1] = 1

        # Third channel will be our food
        for food in game['board']['food']:
            state[food['y'], food['x'], 2] = 1

        return state
    
    def _get_reward(self, game : typ.Dict, current_state : np.ndarray) -> int:
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
        
        head = game['you']['body'][0]
        if self._last_state[head['y'], head['x'], 2] == 1:
            return 10
        
        return -1
    
    def _get_action(self):
        pass

    def _remember(self, state : np.ndarray, action : int, reward : int, next_state : np.ndarray) -> None:
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

        self._memory.append((state, action, reward, next_state))

    def _train_short_memory(self, state : np.ndarray, action : np.ndarray, reward : np.ndarray,  next_state : np.ndarray, done : np.ndarray):
        self._trainer.train_step(state, action, reward, next_state, done)

    def _train_long_memory(self):
        if len(self._memory) > BATCH_SIZE:
            mini_sample = random.sample(self._memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self._memory

        states, actions, rewards, next_states = zip(*mini_sample)
        self._train_short_memory(states, actions, rewards, next_states)

# Start server when `python agent.py` is run
if __name__ == '__main__':
    dqn_agent = Agent()


    from server import run_server

    run_server({'info': dqn_agent.info, 'start': dqn_agent.start, 'move': dqn_agent.move, 'end': dqn_agent.end})
