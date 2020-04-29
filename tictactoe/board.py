import numpy as np
import random


class Board:
    """Represents the tic-tac-toe board. This includes: board display,
    board state calculation, and board updates. 

    Attributes
    ----------
    dim: int
      The dimension of the board. In tic-tac-toe this is a 3x3 board.

    board_space: np.array[np.array[int]]
      The current state of the board.

    current_player: int
      The current player who can make a turn. 1 is for X and 2 is for O.
    """

    def __init__(self):
        self.dim = 3
        self.board_space = np.zeros((self.dim, self.dim), dtype=int)
        self.current_player = 1

    def play_turn(self, state):
        """Sets its board_space to the given state and updates the current player.

        Parameters
        ----------
        state: np.array[np.array[int]]
          The new board state to transition to.
        """
        self.board_space = state
        self.current_player = 1 if self.current_player is 2 else 2

    def calculate_board(self, coord):
        """From the current state and action, return the remaining state of the board
        Parameters
        ----------
        coord: tuple(int,int)
          coordinate representing a potential move of the players

        Return
        ------
        board: np.array[np.array[int]]
          Returns a copy of the board representing the possible state given the action
        """
        board = np.copy(self.board_space)
        x, y = coord
        board[x, y] = self.current_player
        return board

    def generate_slices(self):
        """Takes the current board space and creates all slices of rows, columns, and diagonals.

        Return
        ------
        board_slices: np.array[np.array[int]]
          Slices of the board representing rows, columns, and diagonals
        """
        row_slices = np.copy(self.board_space)
        col_slices = np.transpose(np.copy(self.board_space))
        diag_slices_one = np.diagonal(np.copy(self.board_space))
        diag_slices_two = np.diagonal(
            np.flip(np.copy(self.board_space), axis=0))

        board_slices = np.vstack(
            (row_slices, col_slices, [diag_slices_one], [diag_slices_two]))
        return board_slices

    def end_of_game(self):
        """Determines if the game is over whether by tie or there is a winner

        Return
        ------
        game_over: bool
          boolean to determine if the game is over
        """

        def all_same_player(board_slice):
            """Determines if the slice from the board is all from the same player

                Parameters
                ----------
                board_slice: np.array[int]
                  Slice of the board that is either a row, column, or diagonal represented by integer values of
                  0, 1, or 2 depending on the player moves.
                  If they are all the same, then there is a winner.

                Return
                ------
                : bool
                  Indicates if there is a winner by returning True or False
                """
            return np.array_equal(board_slice, [1, 1, 1]) or np.array_equal(board_slice, [2, 2, 2])

        board_slices = self.generate_slices()
        game_over = np.any([all_same_player(s) for s in board_slices])
        return game_over

    def potential_next_states(self):
        """Determines what states can be entered next given the current state of the board
        and the current player. The states returned are represented as a board_state which
        is just a 2D np.array.

        Returns
        -------
        next_states: 3D np.array
          The potential next states we can enter from the current state. Outer dimension
          is all possible states, and the inner two dimensions are the states themselves.
        """
        x_coords, y_coords = np.where(self.board_space == 0)
        next_states = [self.calculate_board(coord)
                       for coord in zip(x_coords, y_coords)]
        return next_states

    def display(self):
        """Displays the current state of the board using ASCII art. 

        X | X | X
        - | - | -
        O | O | O
        - | - | -
        X | X | X

        Returns
        -------
        board_display: str
          The string representation of the board in the given state.
        """
        value_markers = {0: " ", 1: "X", 2: "O"}
        row_separator = "".join(["-", "|", "-", "|", "-"]) + "\n"

        def display_row(row):
            """Displays the contents of a single row separated by |

            X | X | X

            Parameters
            ----------
            row: np.array[int]
              A single row of the overall board.

            Returns
            -------
            : str
              The string representation of the row.
            """
            return "|".join([value_markers[i] for i in row]) + "\n"

        board_display = row_separator.join(
            [display_row(r) for r in self.board_space])
        return board_display

    def fake_game(self):
        while not self.end_of_game():
            print(self.display())
            if self.current_player == 1:
                next_states = self.potential_next_states()
                state = random.choice(next_states)
                self.play_turn(state)
            else:
                x = input("x: ")
                y = input("y: ")
                self.play_turn(self.calculate_board((int(x), int(y))))
