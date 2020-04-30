from board import Board
from agent import Agent


class Game:
    def __init__(self):
        self.board = Board()
        self.agent = Agent()

    def play_a_turn(self):
        if self.board.current_player == 1:
            self.board.play_turn(self.agent.policy_selection(
                self.board.board_space, self.board.potential_next_states()))
        else:
            x = input("X: ")
            y = input("Y: ")
            self.board.play_turn(self.board.calculate_board((int(x), int(y))))

    def play_game(self):
        while not self.board.end_of_game():
            print(self.board.display())
            self.play_a_turn()
