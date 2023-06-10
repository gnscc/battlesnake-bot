import typing as typ
from collections import deque
import numpy as np

MAX_MEMORY = 100_000

class Agent:
    def __init__(self) -> None:
        self._n_games = 0
        self._last_state = None
        self._last_move = None
        self._memory = deque(maxlen=MAX_MEMORY) # popleft()

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
            reward = 0 # get reward
            self._train_short_memory(reward, current_state)
            self._remember(reward, current_state)
        next_move = self._get_action()

        return {'move': next_move}
    
    def end(self, game : typ.Dict):
        print('END')
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
    
    def _get_action():
        pass

    def _remember(self, reward, next_state):
        self._memory((self._last_state, self._last_move, reward, next_state))

    def _train_short_memory(self, reward, next_state):
        pass

    def _train_long_memory():
        pass

# Start server when `python agent.py` is run
if __name__ == '__main__':
    dqn_agent = Agent()


    from server import run_server

    run_server({'info': dqn_agent.info, 'start': dqn_agent.start, 'move': dqn_agent.move, 'end': dqn_agent.end})
