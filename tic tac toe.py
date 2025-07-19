import math

# Display the board
def print_board(board):
    for row in board:
        print("| " + " | ".join(row) + " |")
    print()

# Check for winner or draw
def check_winner(board, player):
    # Check rows, columns, and diagonals
    for i in range(3):
        if all([cell == player for cell in board[i]]):
            return True
        if all([board[j][i] == player for j in range(3)]):
            return True

    if all([board[i][i] == player for i in range(3)]):
        return True
    if all([board[i][2 - i] == player for i in range(3)]):
        return True

    return False

def is_draw(board):
    return all(cell != ' ' for row in board for cell in row)

# Get available moves
def get_available_moves(board):
    moves = []
    for i in range(3):
        for j in range(3):
            if board[i][j] == ' ':
                moves.append((i, j))
    return moves

# Minimax with Alpha-Beta pruning
def minimax(board, depth, is_maximizing, alpha, beta):
    if check_winner(board, 'O'):
        return 1
    elif check_winner(board, 'X'):
        return -1
    elif is_draw(board):
        return 0

    if is_maximizing:
        max_eval = -math.inf
        for (i, j) in get_available_moves(board):
            board[i][j] = 'O'
            eval = minimax(board, depth + 1, False, alpha, beta)
            board[i][j] = ' '
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = math.inf
        for (i, j) in get_available_moves(board):
            board[i][j] = 'X'
            eval = minimax(board, depth + 1, True, alpha, beta)
            board[i][j] = ' '
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval

# Get the best move for AI
def get_best_move(board):
    best_score = -math.inf
    best_move = None
    for (i, j) in get_available_moves(board):
        board[i][j] = 'O'
        score = minimax(board, 0, False, -math.inf, math.inf)
        board[i][j] = ' '
        if score > best_score:
            best_score = score
            best_move = (i, j)
    return best_move

# Main game loop
def play_game():
    board = [[' ' for _ in range(3)] for _ in range(3)]
    print("Welcome to Tic-Tac-Toe!")
    print("You are X, AI is O.")
    print_board(board)

    while True:
        # Human move
        while True:
            try:
                row = int(input("Enter row (0, 1, 2): "))
                col = int(input("Enter column (0, 1, 2): "))
                if board[row][col] == ' ':
                    board[row][col] = 'X'
                    break
                else:
                    print("Cell already occupied. Try again.")
            except (ValueError, IndexError):
                print("Invalid input. Enter row and column between 0 and 2.")

        print_board(board)

        if check_winner(board, 'X'):
            print("Congratulations! You win!")
            break
        if is_draw(board):
            print("It's a draw!")
            break

        # AI move
        print("AI's turn...")
        ai_move = get_best_move(board)
        board[ai_move[0]][ai_move[1]] = 'O'
        print_board(board)

        if check_winner(board, 'O'):
            print("AI wins! Better luck next time.")
            break
        if is_draw(board):
            print("It's a draw!")
            break

# Run the game
# Run the game
if __name__ == "__main__":
    play_game()