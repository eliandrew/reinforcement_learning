import copy

class Board:

    def __init__(self):
        self.dim = 3
        self.board_space = [[0]*self.dim]*self.dim
        self.current_player = 1

    def update_board(self, coord)
        """From the current state and action, return the remaining state of the board
        Parameters
        ----------
        coord: tuple(int,int)
          coordinate representing a potential move of the players

        Return
        ------
        board: list[list[int]]
          Returns a copy of the board representing the possible state given the action
        """
        board = self.board_space

    def end_of_game(self)
        """Determines if the game is over whether by tie or there is a winner

        Return
        ------
        game_over: bool
          boolean to determine if the game is over 
        """
        def generate_slices(self):
          """Takes the current board space and creates all slices of rows, columns, and diagonals.

          Return
          ------
          board_slices: list[list[int]]
            Slices of the board representing rows, columns, and diagonals 
          """
          board_slices = self.board_space
          for i in self.dim:
            for j in self.dim:
              col.append(self.board_space[j][i])
              if i == j:
                diag.append(self.board_space[i][j])
              elif 
            board_slices.append(col)

        def all_same_player(board_slice)
        """Determines if the slice from the board is all from the same player 

        Parameters
        ----------
        board_slice: list[int]
          Slice of the board that is either a row, column, or diagonal represented by integer values of
          0, 1, or 2 depending on the player moves.
          If they are all the same, then there is a winner.
        
        Return
        ------
        : bool
          Indicates if there is a winner by returning True or False
        """
          return board_slice == [1, 1, 1] or board_slice == [2, 2, 2]

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
            row: list[int]
              A single row of the overall board.

            Returns
            -------
            : str
              The string representation of the row.
            """
            return "|".join([value_markers[i] for i in row]) + "\n"

        board_display = row_separator.join([display_row(r) for r in self.board_space])
        return board_display
