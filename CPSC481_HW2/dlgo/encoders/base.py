import importlib


class Encoder:
    # Lets us support logging or saving the name of the encoder our model is using.
    def name(self):
        raise NotImplementedError()

    # Turn a Go board into a numeric data.
    def encode(self, game_state):
        raise NotImplementedError()

    # Turn a Go board point into an integer index.
    def encode_point(self, point):
        raise NotImplementedError()

    # Turn an integer index back into a Go board point.
    def decode_point_index(self, index):
        raise NotImplementedError()

    # Number of points on the board, i.e. board width times board height.
    def num_points(self):
        raise NotImplementedError()

    # Shape of the encoded board structure.
    def shape(self):
        raise NotImplementedError()


def get_encoder_by_name(name, board_size):  # We can create encoder instances by referencing their name
    if isinstance(board_size, int):
        board_size = (board_size, board_size)  # If board_size is one integer, we create a square board from it.
    module = importlib.import_module('dlgo.encoders.' + name)
    # Each encoder implementation will have to provide a "create" function that provides an instance
    constructor = getattr(module, 'create')
    return constructor(board_size)
