import os
import time

import gurobipy as gp
import numpy as np
import pandas as pd
from numpy import fliplr, rot90


def get_placed_pieces(placements, available_pieces):
    placed_pieces = []
    for i, piece in enumerate(available_pieces):
        for row in range(placements.shape[1]):
            for col in range(placements.shape[2]):
                if placements[i][row][col] is not None and placements[i][row][col].X:
                    placed_pieces.append((piece, (row, col)))
    return placed_pieces


pieces = []
piece_single = np.array([[1]])
piece_double = np.array([[1, 1], [1, 1]])
pieces.extend([piece_single, piece_double])

pieces.extend(
    [
        np.repeat(piece_single, 2, axis=1),
        np.repeat(piece_single, 3, axis=1),
        np.repeat(piece_single, 4, axis=1),
    ]
)
pieces.extend(
    [
        rot90(np.repeat(piece_single, 2, axis=1)),
        rot90(np.repeat(piece_single, 3, axis=1)),
        rot90(np.repeat(piece_single, 4, axis=1)),
    ]
)

piece_l = np.array([[1, 0], [1, 1]])
pieces.extend([piece_l, rot90(piece_l), rot90(piece_l, 3), rot90(piece_l, 2)])

piece_L = np.array([[1, 0, 0], [1, 1, 1]])
pieces.extend([piece_L, fliplr(piece_L), rot90(fliplr(piece_L), 2), rot90(piece_L, 2)])
pieces.extend(
    [
        rot90(fliplr(piece_L), 3),
        rot90(piece_L),
        rot90(piece_L, 3),
        rot90(fliplr(piece_L)),
    ]
)

piece_S = np.array([[0, 1, 1], [1, 1, 0]])
pieces.extend([piece_S, fliplr(piece_S), rot90(piece_S, 1), rot90(fliplr(piece_S))])

piece_T = np.array([[0, 1, 0], [1, 1, 1]])
pieces.extend([piece_T, rot90(piece_T, 2), rot90(piece_T, 3), rot90(piece_T, 1)])


def convert_to_int(x):
    return int(x) if pd.notnull(x) else x


def wait_for_file_change(file_path, timeout=None):
    initial_time = os.path.getmtime(file_path)
    while True:
        current_time = os.path.getmtime(file_path)
        if current_time != initial_time:
            print(f"File '{file_path}' has been modified.")
            return True

        if timeout is not None and time.time() - initial_time >= timeout:
            print("Timeout reached.")
            return False
        time.sleep(0.1)


def decode_board_full(encoded_board_df: pd.DataFrame):
    one_means_white = int(encoded_board_df.iloc[1, -20]) == 1
    board = encoded_board_df.iloc[3:23, -20:].reset_index(drop=True)
    board = np.array(board, dtype=np.bool_)
    if one_means_white:
        board = ~board
    return board


def decode_board(encoded_board_df: pd.DataFrame):
    encoded_board_df.fillna(0, inplace=True)
    column_width = int(encoded_board_df.iloc[29, 0])
    row_width = int(encoded_board_df.iloc[29, 1])
    if column_width == 0 and row_width == 0:
        return decode_board_full(encoded_board_df)
    encoded_board_df = encoded_board_df.iloc[29:].reset_index(drop=True)
    encoded_board = encoded_board_df.iloc[1:, 0].astype(np.int64).tolist()
    board = []
    color = True  # Start with black cells
    for count in encoded_board:
        board.extend([color] * int(count))
        color = not color
    if column_width != 0:
        return np.array(board, dtype=np.bool_).reshape(-1, column_width)
    else:
        return np.array(board, dtype=np.bool_).reshape(row_width, -1, order="F")


def secret_tweaks(model: gp.Model):
    model.setParam("OutputFlag", 0)
    model.setParam("Presolve", 0)
    model.setParam("PoolSolutions", 1)
    model.setParam("MIPFocus", 1)  # Focus on finding feasible solutions quickly
    model.setParam(
        "TimeLimit", 3600
    )  # Solve for a maximum of 3600 seconds (adjust as needed)
    model.setParam("MIPGap", 0.01)  # Set a 1% optimality gap tolerance
    model.setParam("Threads", 16)  # Use 16 threads for parallel solving
    model.setParam("OutputFlag", 1)


PATH = "fast_solution.xlsx"

wait_for_file_change(PATH)

# read xlsx
solution_sheet_name = "Solution"

input_df = pd.read_excel(PATH, sheet_name="Input", header=None)

board = decode_board(input_df)

counts = input_df.iloc[: len(pieces), 0].astype(int).values
available_pieces = [
    item
    for sublist in [[piece] * count for piece, count in zip(pieces, counts)]
    for item in sublist
]

model = gp.Model("doodleFit")
placements = np.full(
    (len(available_pieces), board.shape[0], board.shape[1]), None, dtype=object
)
[
    [
        [
            placements.__setitem__(
                (i, row, col),
                model.addVar(vtype=gp.GRB.BINARY, name=f"pattern_{i}_{row}_{col}"),
            )
            if ~np.any(
                (board[row : row + len(pattern), col : col + len(pattern[0])] == 1)
                & (pattern == 1)
            )
            else None
            for col in range(board.shape[1] - len(pattern[0]) + 1)
        ]
        for row in range(board.shape[0] - len(pattern) + 1)
    ]
    for i, pattern in enumerate(available_pieces)
]

constraints = [
    model.addConstr(
        gp.quicksum(
            [board[i][j]]
            + [
                placements[p][i - ii][j - jj]
                for p in range(len(available_pieces))
                for ii in range(len(available_pieces[p]))
                for jj in range(len(available_pieces[p][0]))
                if available_pieces[p][ii][jj] == 1
                and i - ii >= 0
                and j - jj >= 0
                and placements[p][i - ii][j - jj] is not None
            ]
        )
        <= 1,
        f"no_overlap_{i}_{j}",
    )
    for i in range(board.shape[0])
    for j in range(board.shape[1])
] + [
    model.addConstr(
        gp.quicksum(
            placements[i][row][col]
            for row in range(board.shape[0])
            for col in range(board.shape[1])
            if placements[i][row][col]
        )
        <= 1,
        f"one_placement_piece_{i}",
    )
    for i in range(len(available_pieces))
]

pattern_counts = np.array([np.count_nonzero(pattern) for pattern in available_pieces])
model.setObjective(
    gp.quicksum(
        placements[p][i][j] * pattern_counts[p]
        for p in range(len(available_pieces))
        for i in range(board.shape[0])
        for j in range(board.shape[1])
        if placements[p][i][j] is not None
    )
    + np.count_nonzero(board),
    gp.GRB.MAXIMIZE,
)

model.update()
secret_tweaks(model)
model.optimize()


placed_pieces = get_placed_pieces(placements, available_pieces)

empty_csv = pd.DataFrame(np.full((20, 20), np.nan))
for row in range(board.shape[0]):
    for col in range(board.shape[1]):
        if board[row][col]:
            empty_csv.iloc[row, col] = -1
for idx, piece in enumerate(placed_pieces):
    shape = piece[0]
    for i in range(shape.shape[0]):
        for j in range(shape.shape[1]):
            if shape[i][j]:
                empty_csv.iloc[piece[1][0] + i, piece[1][1] + j] = idx


empty_csv = empty_csv.map(convert_to_int)

empty_csv.to_csv(
    "fast_solution.csv",
    index=False,
    header=False,
)
