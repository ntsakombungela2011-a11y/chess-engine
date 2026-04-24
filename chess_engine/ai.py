"""
Chess AI Module

Implements the chess AI using minimax algorithm with alpha-beta pruning.
The AI evaluates board positions and selects optimal moves.

Classes:
    ChessAI: Main AI class with search and evaluation capabilities
"""

from typing import List, Tuple, Optional
import random

from .pieces import Piece, Position, Pawn, Rook, Knight, Bishop, Queen, King
from .board import Board, Move
from .rules import Rules


class ChessAI:
    """
    Chess AI using minimax with alpha-beta pruning.
    
    The AI evaluates positions using material count and positional bonuses,
    then searches the game tree to find the best move.
    
    Attributes:
        depth: Search depth (default: 3)
        color: The color the AI plays
    """
    
    # Piece values for evaluation
    PIECE_VALUES = {
        "pawn": 100,
        "knight": 320,
        "bishop": 330,
        "rook": 500,
        "queen": 900,
        "king": 20000
    }
    
    # Piece-square tables for positional evaluation
    # Values are from white's perspective; flipped for black
    PAWN_TABLE = [
        [0,  0,  0,  0,  0,  0,  0,  0],
        [50, 50, 50, 50, 50, 50, 50, 50],
        [10, 10, 20, 30, 30, 20, 10, 10],
        [5,  5, 10, 25, 25, 10,  5,  5],
        [0,  0,  0, 20, 20,  0,  0,  0],
        [5, -5,-10,  0,  0,-10, -5,  5],
        [5, 10, 10,-20,-20, 10, 10,  5],
        [0,  0,  0,  0,  0,  0,  0,  0]
    ]
    
    KNIGHT_TABLE = [
        [-50,-40,-30,-30,-30,-30,-40,-50],
        [-40,-20,  0,  0,  0,  0,-20,-40],
        [-30,  0, 10, 15, 15, 10,  0,-30],
        [-30,  5, 15, 20, 20, 15,  5,-30],
        [-30,  0, 15, 20, 20, 15,  0,-30],
        [-30,  5, 10, 15, 15, 10,  5,-30],
        [-40,-20,  0,  5,  5,  0,-20,-40],
        [-50,-40,-30,-30,-30,-30,-40,-50]
    ]
    
    BISHOP_TABLE = [
        [-20,-10,-10,-10,-10,-10,-10,-20],
        [-10,  0,  0,  0,  0,  0,  0,-10],
        [-10,  0,  5, 10, 10,  5,  0,-10],
        [-10,  5,  5, 10, 10,  5,  5,-10],
        [-10,  0, 10, 10, 10, 10,  0,-10],
        [-10, 10, 10, 10, 10, 10, 10,-10],
        [-10,  5,  0,  0,  0,  0,  5,-10],
        [-20,-10,-10,-10,-10,-10,-10,-20]
    ]
    
    ROOK_TABLE = [
        [0,  0,  0,  0,  0,  0,  0,  0],
        [5, 10, 10, 10, 10, 10, 10,  5],
        [-5,  0,  0,  0,  0,  0,  0, -5],
        [-5,  0,  0,  0,  0,  0,  0, -5],
        [-5,  0,  0,  0,  0,  0,  0, -5],
        [-5,  0,  0,  0,  0,  0,  0, -5],
        [-5,  0,  0,  0,  0,  0,  0, -5],
        [0,  0,  0,  5,  5,  0,  0,  0]
    ]
    
    QUEEN_TABLE = [
        [-20,-10,-10, -5, -5,-10,-10,-20],
        [-10,  0,  0,  0,  0,  0,  0,-10],
        [-10,  0,  5,  5,  5,  5,  0,-10],
        [-5,  0,  5,  5,  5,  5,  0, -5],
        [0,  0,  5,  5,  5,  5,  0, -5],
        [-10,  5,  5,  5,  5,  5,  0,-10],
        [-10,  0,  5,  0,  0,  0,  0,-10],
        [-20,-10,-10, -5, -5,-10,-10,-20]
    ]
    
    KING_TABLE_MIDDLE = [
        [-30,-40,-40,-50,-50,-40,-40,-30],
        [-30,-40,-40,-50,-50,-40,-40,-30],
        [-30,-40,-40,-50,-50,-40,-40,-30],
        [-30,-40,-40,-50,-50,-40,-40,-30],
        [-20,-30,-30,-40,-40,-30,-30,-20],
        [-10,-20,-20,-20,-20,-20,-20,-10],
        [20, 20,  0,  0,  0,  0, 20, 20],
        [20, 30, 10,  0,  0, 10, 30, 20]
    ]
    
    def __init__(self, color: str = "black", depth: int = 3):
        """
        Initialize the chess AI.
        
        Args:
            color: The color the AI plays ('white' or 'black')
            depth: Search depth (default: 3)
        """
        if color not in ("white", "black"):
            raise ValueError(f"Invalid color: {color}")
        if depth < 1:
            raise ValueError(f"Depth must be at least 1, got {depth}")
        
        self._color = color
        self._depth = depth
        self._nodes_searched = 0
    
    @property
    def color(self) -> str:
        """Return the AI's color."""
        return self._color
    
    @property
    def depth(self) -> int:
        """Return the search depth."""
        return self._depth
    
    @depth.setter
    def depth(self, value: int) -> None:
        """Set the search depth."""
        if value < 1:
            raise ValueError(f"Depth must be at least 1, got {value}")
        self._depth = value
    
    @property
    def nodes_searched(self) -> int:
        """Return the number of nodes searched in the last search."""
        return self._nodes_searched
    
    def get_best_move(self, board: Board) -> Optional[Move]:
        """
        Find the best move for the current position.
        
        Args:
            board: The current board state
            
        Returns:
            The best move found, or None if no legal moves
        """
        self._nodes_searched = 0
        legal_moves = Rules.get_all_legal_moves(board, self._color)
        
        if not legal_moves:
            return None
        
        # If only one move, return it immediately
        if len(legal_moves) == 1:
            return legal_moves[0]
        
        # Sort moves for better alpha-beta pruning
        legal_moves = self._order_moves(board, legal_moves)
        
        best_move = None
        best_score = float("-inf")
        alpha = float("-inf")
        beta = float("inf")
        
        for move in legal_moves:
            self._nodes_searched += 1
            
            # Make the move on a copy of the board
            temp_board = board.copy()
            temp_board.make_move(move)
            
            # Evaluate the position
            score = self._minimax(temp_board, self._depth - 1, alpha, beta, False)
            
            if score > best_score:
                best_score = score
                best_move = move
            
            alpha = max(alpha, score)
        
        return best_move
    
    def _minimax(self, board: Board, depth: int, alpha: float, beta: float, 
                 is_maximizing: bool) -> float:
        """
        Minimax algorithm with alpha-beta pruning.
        
        Args:
            board: Current board state
            depth: Remaining search depth
            alpha: Best value for maximizer
            beta: Best value for minimizer
            is_maximizing: True if maximizing player's turn
            
        Returns:
            Evaluation score for the position
        """
        # Terminal conditions
        if depth == 0:
            return self._evaluate(board)
        
        if Rules.is_game_over(board):
            return self._evaluate_terminal(board)
        
        color = board.current_turn
        is_ai_turn = (color == self._color)
        
        legal_moves = Rules.get_all_legal_moves(board)
        
        if not legal_moves:
            return self._evaluate_terminal(board)
        
        # Order moves for better pruning
        legal_moves = self._order_moves(board, legal_moves)
        
        if is_maximizing:
            max_eval = float("-inf")
            for move in legal_moves:
                self._nodes_searched += 1
                
                temp_board = board.copy()
                temp_board.make_move(move)
                
                eval_score = self._minimax(temp_board, depth - 1, alpha, beta, False)
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                
                if beta <= alpha:
                    break  # Beta cutoff
            
            return max_eval
        else:
            min_eval = float("inf")
            for move in legal_moves:
                self._nodes_searched += 1
                
                temp_board = board.copy()
                temp_board.make_move(move)
                
                eval_score = self._minimax(temp_board, depth - 1, alpha, beta, True)
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                
                if beta <= alpha:
                    break  # Alpha cutoff
            
            return min_eval
    
    def _order_moves(self, board: Board, moves: List[Move]) -> List[Move]:
        """
        Order moves for better alpha-beta pruning efficiency.
        
        Moves that are likely to be good are searched first:
        1. Captures (especially high-value captures)
        2. Promotions
        3. Castling
        4. Other moves
        
        Args:
            board: Current board state
            moves: List of moves to order
            
        Returns:
            Ordered list of moves
        """
        def move_score(move: Move) -> int:
            score = 0
            
            # Prioritize captures by value
            if move.captured_piece is not None:
                captured_value = self.PIECE_VALUES.get(move.captured_piece.get_type(), 0)
                attacker_value = self.PIECE_VALUES.get(move.piece.get_type(), 0)
                score += 10 * captured_value - attacker_value
            
            # Prioritize promotions
            if move.is_promotion:
                promo_value = self.PIECE_VALUES.get(move.promotion_type, 0)
                score += promo_value
            
            # Slight priority for castling
            if move.is_castling:
                score += 50
            
            # Add some randomness to avoid deterministic play
            score += random.randint(0, 10)
            
            return score
        
        return sorted(moves, key=move_score, reverse=True)
    
    def _evaluate(self, board: Board) -> float:
        """
        Evaluate the current board position.
        
        The evaluation considers:
        1. Material balance
        2. Piece positions
        3. King safety
        4. Pawn structure
        
        Args:
            board: Current board state
            
        Returns:
            Evaluation score (positive = good for AI, negative = bad)
        """
        score = 0
        
        for row in range(8):
            for col in range(8):
                piece = board.get_piece(Position(row=row, col=col))
                if piece is None:
                    continue
                
                piece_type = piece.get_type()
                piece_value = self.PIECE_VALUES.get(piece_type, 0)
                
                # Get positional bonus
                pos_bonus = self._get_positional_bonus(piece, row, col)
                
                if piece.color == self._color:
                    score += piece_value + pos_bonus
                else:
                    score -= (piece_value + pos_bonus)
        
        return score
    
    def _get_positional_bonus(self, piece: Piece, row: int, col: int) -> int:
        """
        Get the positional bonus for a piece at a given square.
        
        Args:
            piece: The piece to evaluate
            row: Row index (0-7)
            col: Column index (0-7)
            
        Returns:
            Positional bonus score
        """
        # Flip row for black pieces (tables are from white's perspective)
        if piece.color == "black":
            row = 7 - row
        
        table = self._get_table_for_piece(piece.get_type())
        if table is None:
            return 0
        
        return table[row][col]
    
    def _get_table_for_piece(self, piece_type: str) -> Optional[List[List[int]]]:
        """Get the piece-square table for a piece type."""
        tables = {
            "pawn": self.PAWN_TABLE,
            "knight": self.KNIGHT_TABLE,
            "bishop": self.BISHOP_TABLE,
            "rook": self.ROOK_TABLE,
            "queen": self.QUEEN_TABLE,
            "king": self.KING_TABLE_MIDDLE
        }
        return tables.get(piece_type)
    
    def _evaluate_terminal(self, board: Board) -> float:
        """
        Evaluate a terminal position (game over).
        
        Args:
            board: Current board state
            
        Returns:
            Large positive/negative score based on result
        """
        if Rules.is_checkmate(board):
            # Checkmate against current turn player
            if board.current_turn == self._color:
                return float("-inf")  # AI lost
            else:
                return float("inf")  # AI won
        
        # Draw conditions
        return 0  # Stalemate or insufficient material
    
    def evaluate_board(self, board: Board) -> float:
        """
        Public method to evaluate a board position.
        
        Args:
            board: Current board state
            
        Returns:
            Evaluation score
        """
        return self._evaluate(board)
