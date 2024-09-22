import numpy as np
import matplotlib.pyplot as plt
import dimod
from dwave.system import DWaveSampler, EmbeddingComposite

# Constants for the game
BOARD_SIZE = 5  # 5x5 board
TIGERS = 3  # Number of tigers in the game
GOATS = 15  # Number of goats to place

# Initialize an empty board: 0 for empty, 1 for goat, -1 for tiger
board = np.zeros((BOARD_SIZE, BOARD_SIZE))

# Tiger positions: We'll place tigers initially
initial_tiger_positions = [(0, 0), (0, BOARD_SIZE-1), (BOARD_SIZE-1, BOARD_SIZE//2)]
for pos in initial_tiger_positions:
    board[pos] = -1

# Display the current board
def display_board(board):
    fig, ax = plt.subplots()
    ax.matshow(board, cmap="coolwarm")
    
    # Add labels for each position
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            if board[i, j] == 1:
                ax.text(j, i, 'G', va='center', ha='center')  # G for Goat
            elif board[i, j] == -1:
                ax.text(j, i, 'T', va='center', ha='center')  # T for Tiger
    
    plt.show()

# Quantum part: QUBO formulation for Tiger's move optimization
def build_qubo_for_tiger_move(board):
    # QUBO dictionary (binary variables representing positions and moves)
    Q = {}
    n = BOARD_SIZE

    # Add potential moves for tigers
    for i in range(n):
        for j in range(n):
            if board[i][j] == -1:  # Tiger
                # Define potential moves (up, down, left, right)
                moves = [(i-2, j), (i+2, j), (i, j-2), (i, j+2)]
                for move in moves:
                    if is_valid_tiger_move(board, (i, j), move):
                        # Assign energy cost (for now we use a placeholder, e.g., -1 to favor moves)
                        Q[((i, j), move)] = -1  # Minimize energy for this move
    return Q

# Function to check if a tiger's move is valid
def is_valid_tiger_move(board, tiger_pos, new_pos):
    x, y = tiger_pos
    new_x, new_y = new_pos
    mid_x, mid_y = (x + new_x) // 2, (y + new_y) // 2
    
    # Ensure the move stays within bounds
    if new_x < 0 or new_y < 0 or new_x >= BOARD_SIZE or new_y >= BOARD_SIZE:
        return False
    
    # Ensure there's a goat to jump over and the new position is empty
    if board[mid_x, mid_y] == 1 and board[new_x, new_y] == 0:
        return True
    return False

# Function to let D-Wave optimize the tiger's move
def optimize_tiger_move(board):
    # Build QUBO model for tiger's next move
    Q = build_qubo_for_tiger_move(board)
    
    if not Q:
        print("No valid moves for tigers!")
        return board
    
    # Solve using D-Wave's quantum annealing
    sampler = EmbeddingComposite(DWaveSampler())
    response = sampler.sample_qubo(Q, num_reads=1000)
    
    # Get the best solution
    best_solution = response.first.sample
    
    # Extract the best move for tigers
    move_from, move_to = None, None
    for key, value in best_solution.items():
        if value == 1:
            move_from, move_to = key
            break
    
    if move_from and move_to:
        print(f"Tiger moves from {move_from} to {move_to}")
        board[move_from] = 0  # Remove tiger from old position
        board[(move_from[0] + move_to[0]) // 2, (move_from[1] + move_to[1]) // 2] = 0  # Capture goat
        board[move_to] = -1  # Place tiger in the new position
    return board

# Function for placing goats (manual/classical)
def place_goat(board, goat_pos):
    x, y = goat_pos
    if board[x, y] == 0:
        board[x, y] = 1  # Place a goat at the given position
    else:
        print("Invalid goat placement!")
    return board

# Game loop
def game_loop():
    goat_count = 0
    
    while goat_count < GOATS:
        display_board(board)
        
        # Goat's turn (place goat manually)
        print(f"Goat's turn. Goats left to place: {GOATS - goat_count}")
        x, y = map(int, input("Enter position to place goat (x y): ").split())
        board = place_goat(board, (x, y))
        goat_count += 1
        
        # Tiger's turn (optimize move using quantum computing)
        print("Tiger's turn...")
        board = optimize_tiger_move(board)
    
    print("Game over!")
    display_board(board)

# Start the game loop
game_loop()
