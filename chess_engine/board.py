"""
Chess Board Module

Provides the board representation and state management for the chess engine.
The board uses an 8x8 matrix internally with Position objects for coordinate access.

Classes:
    Board: Main board class managing piece placement and game state
    Move: Represents a chess move with all necessary metadata
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Any
from copy import deepcopy

from .pieces import (
    Piece, Position, Color, PieceType,
    create_piece, Pawn, Rook, King
)


@dataclass(frozen=True)
class Move:
    """
    Represents a chess move with complete information.
    
    Attributes:
        from_pos: Starting position of the move
        to_pos: Destination position of the move
        piece: The piece being moved
        captured_piece: The piece being captured (if any)
        is_castling: Whether this is a castling move
        is_en_passant: Whether this is an en passant capture
        is_promotion: Whether this move results in pawn promotion
        promotion_type: The type of piece to promote to (default: queen)
    """
    from_pos: Position
    to_pos: Position
    piece: Piece
    captured_piece: Optional[Piece] = None
    is_castling: bool = False
    is_en_passant: bool = False
    is_promotion: bool = False
    promotion_type: PieceType = "queen"
    
    def __hash__(self) -> int:
        return hash((self.from_pos, self.to_pos, self.piece.get_type(), 
                     self.promotion_type))
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Move):
            return NotImplemented
        return (self.from_pos == other.from_pos and
                self.to_pos == other.to_pos and
                self.piece.get_type() == other.piece.get_type() and
                self.promotion_type == other.promotion_type)
    
    def to_algebraic(self) -> str:
        """Convert move to simple algebraic notation."""
        notation = f"{self.from_pos.to_algebraic()}{self.to_pos.to_algebraic()}"
        if self.is_castling:
            if self.to_pos.col > self.from_pos.col:
                return "O-O"  # Kingside
            else:
                return "O-O-O"  # Queenside
        if self.is_promotion:
            notation += self.promotion_type[0].upper()
        return notation
    
    @classmethod
    def from_algebraic(cls, notation: str, board: "Board") -> "Move":
        """
        Create a Move from algebraic notation.
        
        Args:
            notation: Move in algebraic format (e.g., 'e2e4', 'e7e8q')
            board: Current board state
            
        Returns:
            A Move object
            
        Raises:
            ValueError: If notation is invalid or move is illegal
        """
        notation = notation.strip()
        upper_notation = notation.upper()
        
        # Handle castling
        if upper_notation in ("O-O", "0-0"):
            color = board.current_turn
            row = 7 if color == "white" else 0
            from_pos = Position(row=row, col=4)
            to_pos = Position(row=row, col=6)
            piece = board.get_piece(from_pos)
            return cls(from_pos=from_pos, to_pos=to_pos, piece=piece, is_castling=True)
        
        if upper_notation in ("O-O-O", "0-0-0"):
            color = board.current_turn
            row = 7 if color == "white" else 0
            from_pos = Position(row=row, col=4)
            to_pos = Position(row=row, col=2)
            piece = board.get_piece(from_pos)
            return cls(from_pos=from_pos, to_pos=to_pos, piece=piece, is_castling=True)
        
        # Parse standard moves
        notation_lower = notation.lower()
        if len(notation_lower) < 4:
            raise ValueError(f"Invalid move notation: {notation}")
        
        from_pos = Position.from_algebraic(notation_lower[:2])
        to_pos = Position.from_algebraic(notation_lower[2:4])
        
        # Check for promotion
        promotion_type = "queen"
        is_promotion = False
        if len(notation_lower) >= 5:
            promo_char = notation_lower[4].lower()
            promotion_map = {'q': 'queen', 'r': 'rook', 'b': 'bishop', 'n': 'knight'}
            if promo_char in promotion_map:
                promotion_type = promotion_map[promo_char]
                is_promotion = True
        
        piece = board.get_piece(from_pos)
        if piece is None:
            raise ValueError(f"No piece at position {from_pos.to_algebraic()}")
        
        captured_piece = board.get_piece(to_pos)
        
        # Check for en passant
        is_en_passant = (isinstance(piece, Pawn) and 
                        to_pos == board.en_passant_target)
        
        return cls(
            from_pos=from_pos,
            to_pos=to_pos,
            piece=piece,
            captured_piece=captured_piece,
            is_en_passant=is_en_passant,
            is_promotion=is_promotion,
            promotion_type=promotion_type
        )


class Board:
    """
    Chess board representation with complete game state management.
    
    The board uses an 8x8 matrix where each cell contains a Piece object
    or None. Row 0 represents rank 8 (black's back rank) and row 7
    represents rank 1 (white's back rank).
    
    Attributes:
        current_turn: Whose turn it is ('white' or 'black')
        en_passant_target: Position where en passant capture is possible
        halfmove_clock: Counter for 50-move rule
        fullmove_number: Current move number
        move_history: List of all moves made
        castling_rights: Dictionary of castling availability
    """
    
    def __init__(self):
        """Initialize a new chess board with standard starting position."""
        self._board: List[List[Optional[Piece]]] = [[None] * 8 for _ in range(8)]
        self._current_turn: Color = "white"
        self._en_passant_target: Optional[Position] = None
        self._halfmove_clock: int = 0
        self._fullmove_number: int = 1
        self._move_history: List[Move] = []
        self._position_history: List[str] = []
        
        self._setup_initial_position()
    
    def _setup_initial_position(self) -> None:
        """Set up the standard chess starting position."""
        # Place pawns
        for col in range(8):
            self._board[1][col] = Pawn("black")
            self._board[6][col] = Pawn("white")
        
        # Place other pieces
        piece_order: List[PieceType] = ["rook", "knight", "bishop", "queen", 
                                         "king", "bishop", "knight", "rook"]
        
        for col, piece_type in enumerate(piece_order):
            self._board[0][col] = create_piece(piece_type, "black")
            self._board[7][col] = create_piece(piece_type, "white")
    
    def get_piece(self, pos: Position) -> Optional[Piece]:
        """
        Get the piece at the specified position.
        
        Args:
            pos: The position to query
            
        Returns:
            The piece at that position, or None if empty
        """
        if not pos.is_valid():
            return None
        return self._board[pos.row][pos.col]
    
    def set_piece(self, pos: Position, piece: Optional[Piece]) -> None:
        """
        Set a piece at the specified position.
        
        Args:
            pos: The position to set
            piece: The piece to place (or None to clear)
        """
        if not pos.is_valid():
            raise ValueError(f"Invalid position: {pos}")
        self._board[pos.row][pos.col] = piece
    
    def find_king(self, color: Color) -> Optional[Position]:
        """
        Find the king of the specified color.
        
        Args:
            color: The color of the king to find
            
        Returns:
            Position of the king, or None if not found
        """
        for row in range(8):
            for col in range(8):
                piece = self._board[row][col]
                if piece is not None and piece.get_type() == "king" and piece.color == color:
                    return Position(row=row, col=col)
        return None
    
    @property
    def current_turn(self) -> Color:
        """Return whose turn it is."""
        return self._current_turn
    
    @property
    def en_passant_target(self) -> Optional[Position]:
        """Return the en passant target position."""
        return self._en_passant_target
    
    @property
    def halfmove_clock(self) -> int:
        """Return the halfmove clock (for 50-move rule)."""
        return self._halfmove_clock
    
    @property
    def fullmove_number(self) -> int:
        """Return the current move number."""
        return self._fullmove_number
    
    @property
    def move_history(self) -> List[Move]:
        """Return the list of all moves made."""
        return self._move_history.copy()
    
    def get_fen(self) -> str:
        """
        Generate FEN string for current position.
        
        Returns:
            FEN string representation of the current position
        """
        # Piece placement
        rows = []
        for row in range(8):
            empty_count = 0
            row_str = ""
            for col in range(8):
                piece = self._board[row][col]
                if piece is None:
                    empty_count += 1
                else:
                    if empty_count > 0:
                        row_str += str(empty_count)
                        empty_count = 0
                    symbol = piece.get_symbol()
                    # Use standard FEN piece characters
                    fen_chars = {
                        '♙': 'P', '♟': 'p',
                        '♖': 'R', '♜': 'r',
                        '♘': 'N', '♞': 'n',
                        '♗': 'B', '♝': 'b',
                        '♕': 'Q', '♛': 'q',
                        '♔': 'K', '♚': 'k'
                    }
                    row_str += fen_chars.get(symbol, '?')
            if empty_count > 0:
                row_str += str(empty_count)
            rows.append(row_str)
        
        piece_placement = "/".join(rows)
        
        # Castling rights
        castling = ""
        white_king = self.get_piece(Position(row=7, col=4))
        if isinstance(white_king, King) and not white_king.has_moved:
            white_rook_k = self.get_piece(Position(row=7, col=7))
            white_rook_q = self.get_piece(Position(row=7, col=0))
            if isinstance(white_rook_k, Rook) and not white_rook_k.has_moved:
                castling += "K"
            if isinstance(white_rook_q, Rook) and not white_rook_q.has_moved:
                castling += "Q"
        
        black_king = self.get_piece(Position(row=0, col=4))
        if isinstance(black_king, King) and not black_king.has_moved:
            black_rook_k = self.get_piece(Position(row=0, col=7))
            black_rook_q = self.get_piece(Position(row=0, col=0))
            if isinstance(black_rook_k, Rook) and not black_rook_k.has_moved:
                castling += "k"
            if isinstance(black_rook_q, Rook) and not black_rook_q.has_moved:
                castling += "q"
        
        if not castling:
            castling = "-"
        
        # En passant
        ep = self._en_passant_target.to_algebraic() if self._en_passant_target else "-"
        
        # Halfmove and fullmove clocks
        fen = f"{piece_placement} {self._current_turn[0]} {castling} {ep} {self._halfmove_clock} {self._fullmove_number}"
        
        return fen
    
    def make_move(self, move: Move) -> None:
        """
        Execute a move on the board.
        
        This method updates the board state, handles special moves
        (castling, en passant, promotion), and updates game state.
        
        Args:
            move: The move to execute
            
        Raises:
            ValueError: If the move is invalid
        """
        # Validate basic move requirements
        if not move.from_pos.is_valid() or not move.to_pos.is_valid():
            raise ValueError("Invalid move positions")
        
        piece = self.get_piece(move.from_pos)
        if piece is None:
            raise ValueError(f"No piece at {move.from_pos.to_algebraic()}")
        
        if piece.color != self._current_turn:
            raise ValueError(f"It's {self._current_turn}'s turn")
        
        # Store state for undo
        self._position_history.append(self.get_fen())
        
        # Update halfmove clock
        if piece.get_type() == "pawn" or move.captured_piece is not None:
            self._halfmove_clock = 0
        else:
            self._halfmove_clock += 1
        
        # Clear en passant target
        old_ep = self._en_passant_target
        self._en_passant_target = None
        
        # Handle en passant capture
        if move.is_en_passant:
            direction = 1 if piece.color == "white" else -1
            captured_pawn_pos = Position(row=move.to_pos.row + direction, col=move.to_pos.col)
            self.set_piece(captured_pawn_pos, None)
        
        # Set new en passant target (after two-square pawn push)
        if isinstance(piece, Pawn) and abs(move.to_pos.row - move.from_pos.row) == 2:
            ep_row = (move.from_pos.row + move.to_pos.row) // 2
            self._en_passant_target = Position(row=ep_row, col=move.from_pos.col)
        
        # Handle castling
        if move.is_castling:
            row = move.from_pos.row
            if move.to_pos.col == 6:  # Kingside
                rook_from = Position(row=row, col=7)
                rook_to = Position(row=row, col=5)
            else:  # Queenside
                rook_from = Position(row=row, col=0)
                rook_to = Position(row=row, col=3)
            
            rook = self.get_piece(rook_from)
            self.set_piece(rook_from, None)
            self.set_piece(rook_to, rook)
            if rook is not None:
                rook.mark_moved()
        
        # Handle promotion
        if move.is_promotion:
            promoted_piece = create_piece(move.promotion_type, piece.color)
            promoted_piece.mark_moved()  # Promoted pieces have "moved"
            self.set_piece(move.to_pos, promoted_piece)
        else:
            self.set_piece(move.to_pos, piece)
        
        self.set_piece(move.from_pos, None)
        piece.mark_moved()
        
        # Update move history
        self._move_history.append(move)
        
        # Switch turns
        if self._current_turn == "black":
            self._fullmove_number += 1
        self._current_turn = "black" if self._current_turn == "white" else "white"
    
    def undo_move(self) -> Optional[Move]:
        """
        Undo the last move made.
        
        Returns:
            The move that was undone, or None if no moves to undo
        """
        if not self._move_history:
            return None
        
        # Restore previous position from FEN
        if not self._position_history:
            return None
        
        previous_fen = self._position_history.pop()
        self._load_fen(previous_fen)
        
        return self._move_history.pop()
    
    def _load_fen(self, fen: str) -> None:
        """
        Load a position from FEN string.
        
        Args:
            fen: FEN string to load
        """
        parts = fen.split()
        if len(parts) < 6:
            raise ValueError(f"Invalid FEN: {fen}")
        
        # Parse piece placement
        rows = parts[0].split("/")
        self._board = [[None] * 8 for _ in range(8)]
        
        for row_idx, row_str in enumerate(rows):
            col_idx = 0
            for char in row_str:
                if char.isdigit():
                    col_idx += int(char)
                else:
                    color = "white" if char.isupper() else "black"
                    char_lower = char.lower()
                    piece_map = {
                        'p': 'pawn', 'r': 'rook', 'n': 'knight',
                        'b': 'bishop', 'q': 'queen', 'k': 'king'
                    }
                    piece = create_piece(piece_map[char_lower], color)
                    self._board[row_idx][col_idx] = piece
                    col_idx += 1
        
        # Parse other fields
        self._current_turn = "white" if parts[1] == 'w' else "black"
        
        # Parse castling rights (stored in position history, will be recalculated)
        # Parse en passant
        if parts[3] != "-":
            self._en_passant_target = Position.from_algebraic(parts[3])
        else:
            self._en_passant_target = None
        
        self._halfmove_clock = int(parts[4])
        self._fullmove_number = int(parts[5])
    
    def copy(self) -> "Board":
        """
        Create a deep copy of the board.
        
        Returns:
            A new Board instance with identical state
        """
        return deepcopy(self)
    
    def is_empty(self, pos: Position) -> bool:
        """Check if a position is empty."""
        return self.get_piece(pos) is None
    
    def is_enemy(self, pos: Position, color: Color) -> bool:
        """Check if a position contains an enemy piece."""
        piece = self.get_piece(pos)
        return piece is not None and piece.color != color
    
    def is_friendly(self, pos: Position, color: Color) -> bool:
        """Check if a position contains a friendly piece."""
        piece = self.get_piece(pos)
        return piece is not None and piece.color == color
    
    def __repr__(self) -> str:
        """Return ASCII representation of the board."""
        lines = []
        lines.append("  a b c d e f g h")
        lines.append("  ----------------")
        
        for row in range(8):
            rank = f"{8 - row}|"
            for col in range(8):
                piece = self._board[row][col]
                if piece is None:
                    rank += " "
                else:
                    rank += piece.get_symbol()
                rank += " "
            rank += f"|{8 - row}"
            lines.append(rank)
        
        lines.append("  ----------------")
        lines.append("  a b c d e f g h")
        
        return "\n".join(lines)
    
    def display(self) -> None:
        """Print the board to stdout."""
        print(self.__repr__())
