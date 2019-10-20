import argparse
from multiprocessing import freeze_support

import numpy as np

from dlgo.encoders.base import get_encoder_by_name
from dlgo import goboard as goboard
from dlgo import mcts
from dlgo.utils import print_board, print_move


def generate_game(board_size, rounds, max_moves, temperature):
    # In `boards` we store encoded board state, `moves` is for encoded moves.
    boards, moves = [], []

    # We initialize a OnePlaneEncoder by name with given board size.
    encoder = get_encoder_by_name('oneplane', board_size)

    # An new game of size `board_size` is instantiated.
    game = goboard.GameState.new_game(board_size)

    # A Monte Carlo tree search agent with specified number of rounds and temperature will serve as our bot.
    bot = mcts.MCTSAgent(rounds, temperature)

    num_moves = 0
    while not game.is_over():
        print_board(game.board)
        # The next move is selected by the bot.
        move = bot.select_move(game)
        if move.is_play:
            boards.append(encoder.encode(game))  # The encoded board situation is appended to `boards`.

            move_one_hot = np.zeros(encoder.num_points())
            move_one_hot[encoder.encode_point(move.point)] = 1
            moves.append(move_one_hot)  # The one-hot-encoded next move is appended to `moves`.

        print_move(game.next_player, move)
        game = game.apply_move(move)  # Afterwards the bot move is applied to the board.
        num_moves += 1
        if num_moves > max_moves:  # continue with the next move, unless the maximum number of moves has been reached

            break

    return np.array(boards), np.array(moves)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--board-size', '-b', type=int, default=9)
    parser.add_argument('--rounds', '-r', type=int, default=1000)
    parser.add_argument('--temperature', '-t', type=float, default=0.8)
    parser.add_argument('--max-moves', '-m', type=int, default=60,
                        help='Max moves per game.')
    parser.add_argument('--num-games', '-n', type=int, default=10)
    parser.add_argument('--board-out')
    parser.add_argument('--move-out')

    args = parser.parse_args()  # this application allows customization via command-line arguments
    xs = []
    ys = []

    for i in range(args.num_games): # for the specified number of games, you generate game data
        print('Generating game %d/%d...' % (i + 1, args.num_games))
        x, y = generate_game(args.board_size, args.rounds, args.max_moves, args.temperature)
        xs.append(x)
        ys.append(y)

    x = np.concatenate(xs)  # concatenate features and labels after all games have been generated
    y = np.concatenate(ys)

    np.save(args.board_out, x)  # store feature and label data to separate files
    np.save(args.move_out, y)


if __name__ == '__main__':
    freeze_support()
    main()
