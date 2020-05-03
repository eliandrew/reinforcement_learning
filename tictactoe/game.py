from board import Board
from agent import Agent

import numpy as np


class Game:
    def __init__(self):
        self.board = Board()
        self.agent = Agent(self.board)

    def train_and_play_human(self):
        [self.play_agent_game() for i in range(1000)]
        self.play_human_game()

    def play_a_turn_agent(self):
        self.board.play_turn(self.agent.play_turn(
            self.board.board_space, self.board.potential_next_states()))

    def play_agent_game(self):
        while not self.board.end_of_game(self.board.board_space):
            # print(self.board.display())
            # for k, v in self.agent.state_values.items():
            #     if v != 0.5:
            #         print(k, v)
            #     input("Press enter to continue")
            self.play_a_turn_agent()
        print(self.board.display())

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
        while not self.board.end_of_game(self.board.board_space):
            print(self.board.display())
            for k, v in self.agent.state_values.items():
                if v != 0.5:
                    print(k, v)
            self.play_a_turn_human()
        print(self.board.display())
