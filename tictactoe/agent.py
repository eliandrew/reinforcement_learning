from collections import defaultdict
import random
import numpy as np


class Agent:

    def __init__(self):
        self.default_value = 0.0
        self.state_values = defaultdict(lambda: self.default_value)

    def temporal_difference_update(alpha=0.01):
        """Returns the temporal difference update given the current values of the states

        Parameters
        ----------
        alpha: float, optional. Default: 0.01
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
        return self.temporal_difference_update()(self.state_values[start_state], self.state_values[end_state])

    def max_value_action(self, actions):
        """Returns the maximum value from a set of actions
        """
        action_state_values = np.array(
            [self.state_values[a] for a in actions])

        return actions[np.argmax(action_state_values)]

    def epsilon_greedy_policy(self, epsilon=0.01):
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

        return lambda board, actions: random.choices([random.choice(actions), self.max_value_action(actions)], weights=[epsilon, 1-epsilon])

    def policy_selection(self, board_state, actions):
        # , policy_func=lambda: self.epsilon_greedy_policy()):
        """
        """
        return self.epsilon_greedy_policy()(board_state, actions)
