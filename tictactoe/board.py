
class Board:

  @staticmethod
  def display(state):
    """Displays the current state of the board using ASCII art. 

    X | X | X
    - | - | -
    O | O | O
    - | - | -
    X | X | x


    Parameters
    ----------
    state: list[list[int]] 
      A 2D list with dimensions equal to the dimensions of the board.
      Values should be 0 for empty, 1 for player X, or 2 for play O.

    Returns
    -------
    board_display: str
      The string representation of the board in the given state.
    """
    value_markers = {0: " ", 1: "X", 2: "O"}
    row_separator = "".join(["-", "|", "-", "|", "-"]) + "\n"

    def display_row(row):
      return "|".join([value_markers[i] for i in row]) + "\n"

    return row_separator.join([display_row(r) for r in state])

      



