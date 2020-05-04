from board import Board
from agent import Agent

import numpy as np


class Game:
    def __init__(self):
        self.board = Board()
        self.agent = Agent(self.board, 1)
        self.agent_007 = Agent(self.board, 2)

    def train_two_agents(self):
        [self.play_agent_two_game() for i in range(100000)]
        self.play_agent_two_game(True)

    def play_agent_two_game(self, stop_actions=False):
        self.board.reset_board()
        self.agent.epsilon = 0.01
        self.agent_007.epsilon = 0.01
        while (not self.board.end_of_game(self.board.board_space)) and (not self.board.tie_game(self.board.board_space)):
            self.play_a_turn_two_agent(stop_actions)

    def play_a_turn_two_agent(self, stop_actions=False):
        if stop_actions:
            print(self.board.display())
            input("Proceed?")

        if self.board.current_player == 1:
            self.board.play_turn(self.agent.play_turn(
                self.board.board_space, self.board.potential_next_states()))
            board_space = np.copy(self.board.board_space)
            self.agent_007.update_state_values(
                board_space, self.board.board_space)
        else:
            self.board.play_turn(self.agent_007.play_turn(
                self.board.board_space, self.board.potential_next_states()))
            board_space = np.copy(self.board.board_space)
            self.agent.update_state_values(board_space, self.board.board_space)

    def train_and_play_human(self):
        [self.play_agent_game() for i in range(10000)]
        self.play_human_game()

    def play_a_turn_agent(self):
        self.board.play_turn(self.agent.play_turn(
            self.board.board_space, self.board.potential_next_states()))

    def play_agent_game(self):
        self.board.reset_board()
        while (not self.board.end_of_game(self.board.board_space)) and (not self.board.tie_game(self.board.board_space)):
            # print(self.board.display())
            # for k, v in self.agent.state_values.items():
            #     if v != 0.5:
            #         print(k, v)
            #     input("Press enter to continue")

            # print(self.board.tie_game(self.board.board_space))
            # print(self.board.display())
            self.play_a_turn_agent()
        # print(self.board.display())

    def play_a_turn_human(self):
        if self.board.current_player == 1:
            self.board.play_turn(self.agent.play_turn(
                self.board.board_space, self.board.potential_next_states()))
        else:
            x = input("X: ")
            y = input("Y: ")
            board_space = np.copy(self.board.board_space)
            self.board.play_turn(self.board.calculate_board((int(x), int(y))))
            self.agent.update_state_values(board_space, self.board.board_space)

    def play_human_game(self):
        self.board.reset_board()
        self.agent.epsilon = 0.01
        while (not self.board.end_of_game(self.board.board_space)) and (not self.board.tie_game(self.board.board_space)):
            print(self.board.display())
            # for k, v in self.agent.state_values.items():
            #     if v != 0.5:
            #         print(k, v)
            self.play_a_turn_human()
        print(self.board.display())
