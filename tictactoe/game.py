from board import Board
from agent import Agent

import numpy as np


class Game:
    def __init__(self):
        self.board = Board()
        self.agent = Agent(self.board)

    def play_a_turn(self):
        if self.board.current_player == 1:
            self.board.play_turn(self.agent.play_turn(
                self.board.board_space, self.board.potential_next_states()))
        else:
            x = input("X: ")
            y = input("Y: ")
            board_space = np.copy(self.board.board_space)
            self.board.play_turn(self.board.calculate_board((int(x), int(y))))
            self.agent.update_state_values(board_space, self.board.board_space)

    def play_game(self):
        self.board.reset_board()
        while not self.board.end_of_game(self.board.board_space):
            print(self.board.display())
            for k, v in self.agent.state_values.items():
                print(k, v)
            self.play_a_turn()
        print(self.board.display())
