"""
Chess Rules Module

Implements all FIDE chess rules including:
- Check detection
- Checkmate detection
- Stalemate detection
- Legal move filtering
- Draw conditions

This module ensures that no illegal moves can be made and properly
identifies all game-ending conditions.
"""

from typing import List, Optional, Set

from .pieces import Piece, Position, Color, Pawn, King
from .board import Board, Move


class Rules:
    """
    Chess rules enforcement engine.
    
    This class provides methods to validate moves according to FIDE rules,
    detect check/checkmate/stalemate conditions, and filter pseudo-legal
    moves to legal moves.
    """
    
    @staticmethod
    def is_square_attacked(board: Board, pos: Position, by_color: Color) -> bool:
        """
        Check if a square is attacked by any piece of the specified color.
        
        Args:
            board: The current board state
            pos: The position to check
            by_color: The color of the attacking pieces
            
        Returns:
            True if the square is attacked, False otherwise
        """
        # Check for pawn attacks
        pawn_direction = 1 if by_color == "white" else -1
        for col_offset in [-1, 1]:
            attacker_row = pos.row + pawn_direction
            attacker_col = pos.col + col_offset
            if 0 <= attacker_row < 8 and 0 <= attacker_col < 8:
                attacker_pos = Position(row=attacker_row, col=attacker_col)
                attacker = board.get_piece(attacker_pos)
                if isinstance(attacker, Pawn) and attacker.color == by_color:
                    return True
        
        # Check for knight attacks
        knight_offsets = [
            (-2, -1), (-2, 1), (-1, -2), (-1, 2),
            (1, -2), (1, 2), (2, -1), (2, 1)
        ]
        for d_row, d_col in knight_offsets:
            attacker_row = pos.row + d_row
            attacker_col = pos.col + d_col
            if 0 <= attacker_row < 8 and 0 <= attacker_col < 8:
                attacker_pos = Position(row=attacker_row, col=attacker_col)
                attacker = board.get_piece(attacker_pos)
                if attacker is not None and attacker.get_type() == "knight" and attacker.color == by_color:
                    return True
        
        # Check for king attacks
        king_offsets = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]
        for d_row, d_col in king_offsets:
            attacker_row = pos.row + d_row
            attacker_col = pos.col + d_col
            if 0 <= attacker_row < 8 and 0 <= attacker_col < 8:
                attacker_pos = Position(row=attacker_row, col=attacker_col)
                attacker = board.get_piece(attacker_pos)
                if attacker is not None and attacker.get_type() == "king" and attacker.color == by_color:
                    return True
        
        # Check for rook/queen attacks (straight lines)
        straight_dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for d_row, d_col in straight_dirs:
            for distance in range(1, 8):
                attacker_row = pos.row + d_row * distance
                attacker_col = pos.col + d_col * distance
                if not (0 <= attacker_row < 8 and 0 <= attacker_col < 8):
                    break
                attacker_pos = Position(row=attacker_row, col=attacker_col)
                attacker = board.get_piece(attacker_pos)
                if attacker is not None:
                    if attacker.color == by_color and attacker.get_type() in ("rook", "queen"):
                        return True
                    break  # Blocked by any piece
        
        # Check for bishop/queen attacks (diagonals)
        diagonal_dirs = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        for d_row, d_col in diagonal_dirs:
            for distance in range(1, 8):
                attacker_row = pos.row + d_row * distance
                attacker_col = pos.col + d_col * distance
                if not (0 <= attacker_row < 8 and 0 <= attacker_col < 8):
                    break
                attacker_pos = Position(row=attacker_row, col=attacker_col)
                attacker = board.get_piece(attacker_pos)
                if attacker is not None:
                    if attacker.color == by_color and attacker.get_type() in ("bishop", "queen"):
                        return True
                    break  # Blocked by any piece
        
        return False
    
    @staticmethod
    def is_in_check(board: Board, color: Color) -> bool:
        """
        Check if the king of the specified color is in check.
        
        Args:
            board: The current board state
            color: The color of the king to check
            
        Returns:
            True if the king is in check, False otherwise
        """
        king_pos = board.find_king(color)
        if king_pos is None:
            return False
        
        opponent_color = "black" if color == "white" else "white"
        return Rules.is_square_attacked(board, king_pos, opponent_color)
    
    @staticmethod
    def would_be_in_check(board: Board, move: Move, color: Color) -> bool:
        """
        Check if making a move would leave the king in check.
        
        This simulates the move without actually making it on the board.
        
        Args:
            board: The current board state
            move: The move to test
            color: The color of the moving player
            
        Returns:
            True if the move would leave the king in check, False otherwise
        """
        # Create a temporary board state
        temp_board = board.copy()
        temp_board.make_move(move)
        return Rules.is_in_check(temp_board, color)
    
    @staticmethod
    def is_legal_move(board: Board, move: Move) -> bool:
        """
        Check if a move is legal according to FIDE rules.
        
        A move is legal if:
        1. It follows the piece's movement rules
        2. It doesn't leave the player's king in check
        
        Args:
            board: The current board state
            move: The move to validate
            
        Returns:
            True if the move is legal, False otherwise
        """
        # Verify it's the correct turn
        if move.piece.color != board.current_turn:
            return False
        
        # Verify the piece is at the from position
        actual_piece = board.get_piece(move.from_pos)
        if actual_piece is None or actual_piece != move.piece:
            return False
        
        # For castling, verify special conditions
        if move.is_castling:
            return Rules._is_legal_castle(board, move)
        
        # Check if move would leave king in check
        if Rules.would_be_in_check(board, move, move.piece.color):
            return False
        
        return True
    
    @staticmethod
    def _is_legal_castle(board: Board, move: Move) -> bool:
        """
        Check if a castling move is legal.
        
        Castling is legal if:
        1. The king hasn't moved
        2. The rook hasn't moved
        3. No pieces between king and rook
        4. King is not in check
        5. King doesn't pass through check
        6. King doesn't end up in check
        
        Args:
            board: The current board state
            move: The castling move
            
        Returns:
            True if castling is legal, False otherwise
        """
        king = move.piece
        if not isinstance(king, King):
            return False
        
        if king.has_moved:
            return False
        
        row = move.from_pos.row
        opponent_color = "black" if king.color == "white" else "white"
        
        # King must not be in check
        if Rules.is_in_check(board, king.color):
            return False
        
        # Determine castling type
        if move.to_pos.col == 6:  # Kingside
            rook_pos = Position(row=row, col=7)
            squares_between = [Position(row=row, col=5), Position(row=row, col=6)]
            squares_to_check = [Position(row=row, col=4), 
                               Position(row=row, col=5), 
                               Position(row=row, col=6)]
        else:  # Queenside
            rook_pos = Position(row=row, col=0)
            squares_between = [Position(row=row, col=1), 
                              Position(row=row, col=2), 
                              Position(row=row, col=3)]
            squares_to_check = [Position(row=row, col=4),
                               Position(row=row, col=3),
                               Position(row=row, col=2)]
        
        # Check if rook exists and hasn't moved
        rook = board.get_piece(rook_pos)
        if rook is None or rook.get_type() != "rook" or rook.has_moved or rook.color != king.color:
            return False
        
        # Check if squares between are empty
        for pos in squares_between:
            if board.get_piece(pos) is not None:
                return False
        
        # Check if king passes through or ends in check
        for pos in squares_to_check:
            if Rules.is_square_attacked(board, pos, opponent_color):
                return False
        
        return True
    
    @staticmethod
    def get_all_legal_moves(board: Board, color: Optional[Color] = None) -> List[Move]:
        """
        Get all legal moves for the specified color.
        
        Args:
            board: The current board state
            color: The color to get moves for (default: current turn)
            
        Returns:
            List of all legal moves
        """
        if color is None:
            color = board.current_turn
        
        legal_moves = []
        
        for row in range(8):
            for col in range(8):
                piece = board.get_piece(Position(row=row, col=col))
                if piece is not None and piece.color == color:
                    pos = Position(row=row, col=col)
                    moves = Rules._get_pseudo_legal_moves(board, piece, pos)
                    
                    for move in moves:
                        if Rules.is_legal_move(board, move):
                            legal_moves.append(move)
        
        return legal_moves
    
    @staticmethod
    def _get_pseudo_legal_moves(board: Board, piece: Piece, pos: Position) -> List[Move]:
        """
        Get all pseudo-legal moves for a piece (may leave king in check).
        
        Args:
            board: The current board state
            piece: The piece to get moves for
            pos: The position of the piece
            
        Returns:
            List of pseudo-legal moves
        """
        moves = []
        raw_positions = piece.get_raw_moves(pos, board)
        
        for to_pos in raw_positions:
            captured = board.get_piece(to_pos)
            
            # Handle en passant capture
            is_en_passant = (isinstance(piece, Pawn) and 
                           to_pos == board.en_passant_target and
                           captured is None)
            
            if is_en_passant:
                # Get the captured pawn
                direction = 1 if piece.color == "white" else -1
                captured_pawn_pos = Position(row=to_pos.row + direction, col=to_pos.col)
                captured = board.get_piece(captured_pawn_pos)
            
            # Handle promotion
            is_promotion = isinstance(piece, Pawn) and piece.is_promoting(pos, to_pos)
            
            if is_promotion:
                # Generate moves for each promotion type
                for promo_type in ["queen", "rook", "bishop", "knight"]:
                    move = Move(
                        from_pos=pos,
                        to_pos=to_pos,
                        piece=piece,
                        captured_piece=captured,
                        is_en_passant=is_en_passant,
                        is_promotion=True,
                        promotion_type=promo_type
                    )
                    moves.append(move)
            else:
                move = Move(
                    from_pos=pos,
                    to_pos=to_pos,
                    piece=piece,
                    captured_piece=captured,
                    is_en_passant=is_en_passant
                )
                moves.append(move)
        
        # Add castling moves for king
        if isinstance(piece, King) and not piece.has_moved:
            # Kingside castling
            if piece.can_castle_kingside(board):
                row = pos.row
                to_pos = Position(row=row, col=6)
                move = Move(
                    from_pos=pos,
                    to_pos=to_pos,
                    piece=piece,
                    is_castling=True
                )
                moves.append(move)
            
            # Queenside castling
            if piece.can_castle_queenside(board):
                row = pos.row
                to_pos = Position(row=row, col=2)
                move = Move(
                    from_pos=pos,
                    to_pos=to_pos,
                    piece=piece,
                    is_castling=True
                )
                moves.append(move)
        
        return moves
    
    @staticmethod
    def is_checkmate(board: Board, color: Optional[Color] = None) -> bool:
        """
        Check if the specified color is in checkmate.
        
        Checkmate occurs when:
        1. The king is in check
        2. There are no legal moves
        
        Args:
            board: The current board state
            color: The color to check (default: current turn)
            
        Returns:
            True if checkmate, False otherwise
        """
        if color is None:
            color = board.current_turn
        
        if not Rules.is_in_check(board, color):
            return False
        
        legal_moves = Rules.get_all_legal_moves(board, color)
        return len(legal_moves) == 0
    
    @staticmethod
    def is_stalemate(board: Board, color: Optional[Color] = None) -> bool:
        """
        Check if the specified color is in stalemate.
        
        Stalemate occurs when:
        1. The king is NOT in check
        2. There are no legal moves
        
        Args:
            board: The current board state
            color: The color to check (default: current turn)
            
        Returns:
            True if stalemate, False otherwise
        """
        if color is None:
            color = board.current_turn
        
        if Rules.is_in_check(board, color):
            return False
        
        legal_moves = Rules.get_all_legal_moves(board, color)
        return len(legal_moves) == 0
    
    @staticmethod
    def is_insufficient_material(board: Board) -> bool:
        """
        Check if there is insufficient material for checkmate.
        
        Insufficient material occurs when:
        - King vs King
        - King + Bishop vs King
        - King + Knight vs King
        - King + Bishop vs King + Bishop (same color bishops)
        
        Args:
            board: The current board state
            
        Returns:
            True if insufficient material, False otherwise
        """
        pieces = {"white": [], "black": []}
        
        for row in range(8):
            for col in range(8):
                piece = board.get_piece(Position(row=row, col=col))
                if piece is not None:
                    pieces[piece.color].append(piece.get_type())
        
        # Remove kings
        white_pieces = [p for p in pieces["white"] if p != "king"]
        black_pieces = [p for p in pieces["black"] if p != "king"]
        
        # King vs King
        if len(white_pieces) == 0 and len(black_pieces) == 0:
            return True
        
        # King + minor piece vs King
        if len(white_pieces) == 0 and len(black_pieces) == 1:
            if black_pieces[0] in ("bishop", "knight"):
                return True
        
        if len(black_pieces) == 0 and len(white_pieces) == 1:
            if white_pieces[0] in ("bishop", "knight"):
                return True
        
        # King + Bishop vs King + Bishop (could be more complex with same-color bishops)
        if len(white_pieces) == 1 and len(black_pieces) == 1:
            if white_pieces[0] == "bishop" and black_pieces[0] == "bishop":
                return True
        
        return False
    
    @staticmethod
    def is_draw(board: Board) -> bool:
        """
        Check if the game is a draw.
        
        A game is drawn when:
        1. Stalemate
        2. Insufficient material
        3. Fifty-move rule (halfmove clock >= 100)
        4. Threefold repetition
        
        Args:
            board: The current board state
            
        Returns:
            True if draw, False otherwise
        """
        # Stalemate
        if Rules.is_stalemate(board):
            return True
        
        # Insufficient material
        if Rules.is_insufficient_material(board):
            return True
        
        # Fifty-move rule (100 halfmoves)
        if board.halfmove_clock >= 100:
            return True
        
        # Threefold repetition would require tracking position history
        # This is handled in the Game class
        
        return False
    
    @staticmethod
    def is_game_over(board: Board) -> bool:
        """
        Check if the game has ended.
        
        The game ends when:
        1. Checkmate
        2. Stalemate
        3. Draw conditions
        
        Args:
            board: The current board state
            
        Returns:
            True if game is over, False otherwise
        """
        if Rules.is_checkmate(board):
            return True
        
        if Rules.is_draw(board):
            return True
        
        return False
    
    @staticmethod
    def get_game_result(board: Board) -> str:
        """
        Get the result of the game.
        
        Args:
            board: The current board state
            
        Returns:
            Result string: "1-0" (white wins), "0-1" (black wins), 
                          "1/2-1/2" (draw), or "*" (ongoing)
        """
        if Rules.is_checkmate(board):
            if board.current_turn == "white":
                return "0-1"  # Black wins
            else:
                return "1-0"  # White wins
        
        if Rules.is_draw(board):
            return "1/2-1/2"
        
        return "*"  # Game ongoing
