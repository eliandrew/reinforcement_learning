from collections import defaultdict
import random
import datetime
import numpy as np


class Agent:
    """Represents the agent playing the tic-tac-toe game. This includes: playing a turn,
        updating the values for the states, and executing the policy.

        Attributes
        ----------
        board: np.array[np.array[]]
            Representation of the board for the agent to reference

        default_value: float
            Parameter to initialize the states values of moves that do not result in termination of the game

        state_values: dict[tuple(tuple(int)), float]
            Initial representation of the state values to be added to and adjusted over time as new states are experienced
    """

    def __init__(self, board):
        self.board = board
        self.default_value = 0.5
        self.state_values = {}
        random.seed(datetime.datetime.now().microsecond)

    def convert_state_to_key(self, state):
        """Converts the given state into a tuple(tuple(int)) that
        can be used as a key in state_values.

        Parameters
        ----------
        state: np.array[np.array[int]]
            The state of the board that we want to convert.

        Returns
        -------
        key: tuple(tuple(int))
            2D tuple that represents the state and can be used
            as a key in a dictionary.
        """
        key = tuple(tuple(r) for r in state)
        return key

    def value_for_state(self, state):
        """Looks up the current value of the given state by
        first converting to the appropriate key and then
        performing a lookup.

        Parameters
        ----------
        state: np.array[np.array[int]]
            The board state for which we want to lookup the
            state value.

        Returns
        -------
        : float
            The value of the given state.
        """

        key = self.convert_state_to_key(state)
        if key not in self.state_values:
            game_value = self.board.end_of_game(state)
            if game_value == 0:
                self.state_values[key] = 0.5
            elif game_value == 1:
                self.state_values[key] = 1.0
            else:
                self.state_values[key] = 0.0

        return self.state_values[key]

    def temporal_difference_update(self, alpha=0.8):
        """Returns the temporal difference update given the current values of the states

        Parameters
        ----------
        alpha: float, optional. Default: 0.8
            learning rate of the temporal differece algorithm

        Return
        ------
        : function(float, float)
            Function calculating the temporal difference given the alpha value

        value_one: float
            Value of the state that is being modified

        value_two: float
            Value of the state that the modified state is approaching
        """

        return lambda value_one, value_two: value_one + alpha * (value_two - value_one)

    def calculate_value_update(self, start_state, end_state):
        # , update_func=lambda: self.temporal_difference_update()):
        """Calculate the updates to the values from the start_state to the end_state

        Parameters
        ----------
        start_state: np.array[np.array[int]]
            The initial state of the board used to calculate the next move

        end_state: np.array[np.array[int]]
            The state of the board after the move that is being considered is taken

        update_func: function(float, float), optional. Default: temporal_difference_update
            A function that implements an update algorithm for the state values

        Returns
        -------
        : float
            Updated state value for the start state

        """
        return self.temporal_difference_update()(self.value_for_state(start_state), self.value_for_state(end_state))

    def update_state_values(self, current_state, next_state):
        """Updates the state_values using the given current_state
        and next_state.

        Parameters
        ----------
        current_state: np.array[np.array[int]]
            The current state of the board where the agent is moving from.

        next_state: np.array[np.array[int]]
            The next state of the board where the agent is moving to.
        """
        self.state_values[self.convert_state_to_key(
            current_state)] = self.calculate_value_update(current_state, next_state)

    def max_value_action(self, actions):
        """Returns the maximum value from a set of actions
        """
        action_state_values = np.array(
            [self.value_for_state(a) for a in actions])

        return actions[np.argmax(action_state_values)]

    def epsilon_greedy_policy(self, epsilon=0.5):
        """Returns the epsilon greedy algorithm given the epsilon value

         Parameters
         ----------
         epsilon: float, optional. Default: 0.01
             Frequency of exploration

         Return
         ------
         : function(np.array[np.array[int]], np.array[np.array[int]])
             A function where the frequency of the highest valued option is returned with
             a 1-epsilon frequency, and a random choice among the other options is returned
             with an epsilon frequency

         board: np.array[np.array[int]]
             The current state of the board

         states: np.array[np.array[int]]
             The potential actions to be taken
         """

        random_choice_weights = []
        [random_choice_weights.append(1/(len(actions))) for i in len(actions)]
        random_choices = (random.choices(
            [actions], weights=random_choice_weights))
        return lambda board, actions: random.choices([[random_choices], self.max_value_action(actions)], weights=[epsilon, 1-epsilon])[0]

    def policy_selection(self, board_state, actions):
        # , policy_func=lambda: self.epsilon_greedy_policy()):
        """Selects the policy to adhere to based on the current board and the possible actions

        Returns
        -------
        :
        """
        return self.epsilon_greedy_policy()(board_state, actions)

    def play_turn(self, current_state, next_states):
        """Plays a single turn for the agent given the current_state
        and potential next_states. This includes finding the optimal action to take
        and then updating the appropriate state values.

        Parameters
        ----------
        current_state: np.array[np.array[int]]
            The current state of the board where the agent is moving from.

        next_states: list[np.array[np.array[int]]]
            The potential next states of the board where the agent can move to.

        Returns
        -------
        next_state: np.array[np.array[int]]
            The next state that the agent is choosing to move to.
        """
        next_state = self.policy_selection(current_state, next_states)
        self.update_state_values(current_state, next_state)
        return next_state
