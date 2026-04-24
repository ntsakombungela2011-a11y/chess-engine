"""
Chess Pieces Module

Defines all chess pieces with their movement rules according to FIDE standards.
Each piece knows its valid movement patterns, but legality is determined by
the board and rules modules.

Classes:
    Piece: Base class for all chess pieces
    Pawn: Pawn piece with special promotion and en passant rules
    Rook: Rook piece with castling capability
    Knight: Knight piece with L-shaped movement
    Bishop: Bishop piece with diagonal movement
    Queen: Queen piece combining rook and bishop movement
    King: King piece with castling capability
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .board import Board


@dataclass(frozen=True)
class Position:
    """
    Represents a position on the chess board.
    
    Attributes:
        row: Row index (0-7, where 0 is rank 8 and 7 is rank 1)
        col: Column index (0-7, where 0 is file a and 7 is file h)
    """
    row: int
    col: int
    
    def __post_init__(self):
        if not (0 <= self.row < 8):
            raise ValueError(f"Row must be 0-7, got {self.row}")
        if not (0 <= self.col < 8):
            raise ValueError(f"Column must be 0-7, got {self.col}")
    
    def is_valid(self) -> bool:
        """Check if position is within board bounds."""
        return 0 <= self.row < 8 and 0 <= self.col < 8
    
    def to_algebraic(self) -> str:
        """Convert to algebraic notation (e.g., 'e4')."""
        return f"{chr(ord('a') + self.col)}{8 - self.row}"
    
    @classmethod
    def from_algebraic(cls, notation: str) -> "Position":
        """Create position from algebraic notation (e.g., 'e4')."""
        if len(notation) != 2:
            raise ValueError(f"Invalid algebraic notation: {notation}")
        col = ord(notation[0].lower()) - ord('a')
        row = 8 - int(notation[1])
        return cls(row=row, col=col)


Color = str  # Type alias: 'white' or 'black'
PieceType = str  # Type alias: 'pawn', 'rook', 'knight', 'bishop', 'queen', 'king'


class Piece(ABC):
    """
    Abstract base class for all chess pieces.
    
    Attributes:
        color: The color of the piece ('white' or 'black')
        has_moved: Whether the piece has moved (important for castling)
    """
    
    def __init__(self, color: Color):
        """
        Initialize a chess piece.
        
        Args:
            color: The color of the piece ('white' or 'black')
        """
        if color not in ('white', 'black'):
            raise ValueError(f"Invalid color: {color}")
        self._color = color
        self._has_moved = False
    
    @property
    def color(self) -> Color:
        """Return the color of the piece."""
        return self._color
    
    @property
    def has_moved(self) -> bool:
        """Return whether the piece has moved."""
        return self._has_moved
    
    def mark_moved(self) -> None:
        """Mark the piece as having moved."""
        self._has_moved = True
    
    @abstractmethod
    def get_type(self) -> PieceType:
        """Return the type of the piece."""
        pass
    
    @abstractmethod
    def get_symbol(self) -> str:
        """Return the Unicode symbol for the piece."""
        pass
    
    @abstractmethod
    def get_raw_moves(self, pos: Position, board: "Board") -> List[Position]:
        """
        Get all positions this piece can move to (ignoring check).
        
        This returns pseudo-legal moves that may leave the king in check.
        The rules module will filter these to legal moves.
        
        Args:
            pos: Current position of the piece
            board: The current board state
            
        Returns:
            List of positions the piece can move to
        """
        pass
    
    def __repr__(self) -> str:
        return f"{self.color} {self.get_type()}"
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Piece):
            return NotImplemented
        return self.color == other.color and self.get_type() == other.get_type()


class Pawn(Piece):
    """
    Pawn piece with standard movement and capture rules.
    
    Pawns move forward one square but capture diagonally.
    On first move, they can move two squares forward.
    They can capture en passant and promote on reaching the last rank.
    """
    
    def get_type(self) -> PieceType:
        return "pawn"
    
    def get_symbol(self) -> str:
        return "♙" if self.color == "white" else "♟"
    
    def get_raw_moves(self, pos: Position, board: "Board") -> List[Position]:
        moves = []
        direction = -1 if self.color == "white" else 1
        start_row = 6 if self.color == "white" else 1
        
        # Forward move (one square)
        new_row = pos.row + direction
        if 0 <= new_row < 8:
            new_pos = Position(row=new_row, col=pos.col)
            if board.get_piece(new_pos) is None:
                moves.append(new_pos)
                
                # Forward move (two squares from starting position)
                if pos.row == start_row and not self.has_moved:
                    double_row = pos.row + 2 * direction
                    if 0 <= double_row < 8:
                        double_pos = Position(row=double_row, col=pos.col)
                        if board.get_piece(double_pos) is None:
                            moves.append(double_pos)
        
        # Diagonal captures
        for col_offset in [-1, 1]:
            new_col = pos.col + col_offset
            new_row = pos.row + direction
            if 0 <= new_row < 8 and 0 <= new_col < 8:
                capture_pos = Position(row=new_row, col=new_col)
                target = board.get_piece(capture_pos)
                if target is not None and target.color != self.color:
                    moves.append(capture_pos)
                
                # En passant capture
                elif board.en_passant_target == capture_pos:
                    moves.append(capture_pos)
        
        return moves
    
    def get_promotion_positions(self, pos: Position) -> List[Position]:
        """
        Get positions where this pawn would promote.
        
        Args:
            pos: Current position of the pawn
            
        Returns:
            List of positions where promotion occurs
        """
        direction = -1 if self.color == "white" else 1
        promotion_row = 0 if self.color == "white" else 7
        
        promotions = []
        for col_offset in [-1, 0, 1]:
            new_col = pos.col + col_offset
            if 0 <= new_col < 8:
                promo_pos = Position(row=promotion_row, col=new_col)
                if promo_pos.is_valid():
                    promotions.append(promo_pos)
        
        return promotions
    
    def is_promoting(self, from_pos: Position, to_pos: Position) -> bool:
        """Check if this move results in promotion."""
        promotion_row = 0 if self.color == "white" else 7
        return to_pos.row == promotion_row


class Rook(Piece):
    """
    Rook piece with horizontal and vertical movement.
    
    Rooks move any number of squares horizontally or vertically.
    They can castle with the king under specific conditions.
    """
    
    def get_type(self) -> PieceType:
        return "rook"
    
    def get_symbol(self) -> str:
        return "♖" if self.color == "white" else "♜"
    
    def get_raw_moves(self, pos: Position, board: "Board") -> List[Position]:
        moves = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, down, left, right
        
        for d_row, d_col in directions:
            for distance in range(1, 8):
                new_row = pos.row + d_row * distance
                new_col = pos.col + d_col * distance
                if not (0 <= new_row < 8 and 0 <= new_col < 8):
                    break
                
                new_pos = Position(row=new_row, col=new_col)
                target = board.get_piece(new_pos)
                if target is None:
                    moves.append(new_pos)
                elif target.color != self.color:
                    moves.append(new_pos)
                    break
                else:
                    break
        
        return moves


class Knight(Piece):
    """
    Knight piece with L-shaped movement.
    
    Knights move in an L-shape: two squares in one direction and one
    square perpendicular. They can jump over other pieces.
    """
    
    def get_type(self) -> PieceType:
        return "knight"
    
    def get_symbol(self) -> str:
        return "♘" if self.color == "white" else "♞"
    
    def get_raw_moves(self, pos: Position, board: "Board") -> List[Position]:
        moves = []
        offsets = [
            (-2, -1), (-2, 1), (-1, -2), (-1, 2),
            (1, -2), (1, 2), (2, -1), (2, 1)
        ]
        
        for d_row, d_col in offsets:
            new_row = pos.row + d_row
            new_col = pos.col + d_col
            if 0 <= new_row < 8 and 0 <= new_col < 8:
                new_pos = Position(row=new_row, col=new_col)
                target = board.get_piece(new_pos)
                if target is None or target.color != self.color:
                    moves.append(new_pos)
        
        return moves


class Bishop(Piece):
    """
    Bishop piece with diagonal movement.
    
    Bishops move any number of squares diagonally.
    """
    
    def get_type(self) -> PieceType:
        return "bishop"
    
    def get_symbol(self) -> str:
        return "♗" if self.color == "white" else "♝"
    
    def get_raw_moves(self, pos: Position, board: "Board") -> List[Position]:
        moves = []
        directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        
        for d_row, d_col in directions:
            for distance in range(1, 8):
                new_row = pos.row + d_row * distance
                new_col = pos.col + d_col * distance
                if not (0 <= new_row < 8 and 0 <= new_col < 8):
                    break
                
                new_pos = Position(row=new_row, col=new_col)
                target = board.get_piece(new_pos)
                if target is None:
                    moves.append(new_pos)
                elif target.color != self.color:
                    moves.append(new_pos)
                    break
                else:
                    break
        
        return moves


class Queen(Piece):
    """
    Queen piece combining rook and bishop movement.
    
    Queens can move any number of squares horizontally, vertically, or diagonally.
    """
    
    def get_type(self) -> PieceType:
        return "queen"
    
    def get_symbol(self) -> str:
        return "♕" if self.color == "white" else "♛"
    
    def get_raw_moves(self, pos: Position, board: "Board") -> List[Position]:
        moves = []
        # Combine rook and bishop directions
        directions = [
            (-1, 0), (1, 0), (0, -1), (0, 1),  # Rook directions
            (-1, -1), (-1, 1), (1, -1), (1, 1)  # Bishop directions
        ]
        
        for d_row, d_col in directions:
            for distance in range(1, 8):
                new_row = pos.row + d_row * distance
                new_col = pos.col + d_col * distance
                if not (0 <= new_row < 8 and 0 <= new_col < 8):
                    break
                
                new_pos = Position(row=new_row, col=new_col)
                target = board.get_piece(new_pos)
                if target is None:
                    moves.append(new_pos)
                elif target.color != self.color:
                    moves.append(new_pos)
                    break
                else:
                    break
        
        return moves


class King(Piece):
    """
    King piece with single-square movement in any direction.
    
    Kings move one square in any direction. They can castle with
    a rook under specific conditions.
    """
    
    def get_type(self) -> PieceType:
        return "king"
    
    def get_symbol(self) -> str:
        return "♔" if self.color == "white" else "♚"
    
    def get_raw_moves(self, pos: Position, board: "Board") -> List[Position]:
        moves = []
        directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]
        
        for d_row, d_col in directions:
            new_row = pos.row + d_row
            new_col = pos.col + d_col
            if 0 <= new_row < 8 and 0 <= new_col < 8:
                new_pos = Position(row=new_row, col=new_col)
                target = board.get_piece(new_pos)
                if target is None or target.color != self.color:
                    moves.append(new_pos)
        
        # Castling moves (handled specially in movegen)
        return moves
    
    def can_castle_kingside(self, board: "Board") -> bool:
        """Check if kingside castling is possible (not if path is clear)."""
        if self.has_moved:
            return False
        
        row = 7 if self.color == "white" else 0
        rook_pos = Position(row=row, col=7)
        rook = board.get_piece(rook_pos)
        
        return (rook is not None and 
                rook.get_type() == "rook" and 
                rook.color == self.color and
                not rook.has_moved)
    
    def can_castle_queenside(self, board: "Board") -> bool:
        """Check if queenside castling is possible (not if path is clear)."""
        if self.has_moved:
            return False
        
        row = 7 if self.color == "white" else 0
        rook_pos = Position(row=row, col=0)
        rook = board.get_piece(rook_pos)
        
        return (rook is not None and
                rook.get_type() == "rook" and
                rook.color == self.color and
                not rook.has_moved)


def create_piece(piece_type: PieceType, color: Color) -> Piece:
    """
    Factory function to create a piece of the specified type.
    
    Args:
        piece_type: The type of piece to create
        color: The color of the piece
        
    Returns:
        A new piece instance
        
    Raises:
        ValueError: If piece_type is invalid
    """
    piece_classes = {
        "pawn": Pawn,
        "rook": Rook,
        "knight": Knight,
        "bishop": Bishop,
        "queen": Queen,
        "king": King
    }
    
    if piece_type not in piece_classes:
        raise ValueError(f"Invalid piece type: {piece_type}")
    
    return piece_classes[piece_type](color)
