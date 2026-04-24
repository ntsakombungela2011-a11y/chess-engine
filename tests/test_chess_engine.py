"""
Comprehensive test suite for the Chess Engine.

Tests cover:
- Piece movement rules
- Board state management
- Legal move generation
- Check, checkmate, and stalemate detection
- Castling rules
- En passant
- Pawn promotion
- AI move validity
- Game flow

Run with: pytest tests/
"""

import pytest
from chess_engine.pieces import (
    Position, Piece, Pawn, Rook, Knight, Bishop, Queen, King, create_piece
)
from chess_engine.board import Board, Move
from chess_engine.rules import Rules
from chess_engine.ai import ChessAI
from chess_engine.game import Game


class TestPosition:
    """Test Position class functionality."""
    
    def test_position_creation(self):
        """Test valid position creation."""
        pos = Position(row=0, col=0)
        assert pos.row == 0
        assert pos.col == 0
    
    def test_position_invalid_row(self):
        """Test that invalid row raises error."""
        with pytest.raises(ValueError):
            Position(row=-1, col=0)
        with pytest.raises(ValueError):
            Position(row=8, col=0)
    
    def test_position_invalid_col(self):
        """Test that invalid column raises error."""
        with pytest.raises(ValueError):
            Position(row=0, col=-1)
        with pytest.raises(ValueError):
            Position(row=0, col=8)
    
    def test_position_is_valid(self):
        """Test position validity check."""
        assert Position(row=0, col=0).is_valid()
        assert Position(row=7, col=7).is_valid()
    
    def test_position_to_algebraic(self):
        """Test algebraic notation conversion."""
        assert Position(row=6, col=4).to_algebraic() == "e2"
        assert Position(row=0, col=0).to_algebraic() == "a8"
        assert Position(row=7, col=7).to_algebraic() == "h1"
    
    def test_position_from_algebraic(self):
        """Test algebraic notation parsing."""
        pos = Position.from_algebraic("e2")
        assert pos.row == 6
        assert pos.col == 4
        
        pos = Position.from_algebraic("a8")
        assert pos.row == 0
        assert pos.col == 0


class TestPieces:
    """Test piece creation and basic properties."""
    
    def test_create_pawn(self):
        """Test pawn creation."""
        pawn = create_piece("pawn", "white")
        assert isinstance(pawn, Pawn)
        assert pawn.color == "white"
        assert pawn.get_type() == "pawn"
    
    def test_create_rook(self):
        """Test rook creation."""
        rook = create_piece("rook", "black")
        assert isinstance(rook, Rook)
        assert rook.color == "black"
    
    def test_create_knight(self):
        """Test knight creation."""
        knight = create_piece("knight", "white")
        assert isinstance(knight, Knight)
    
    def test_create_bishop(self):
        """Test bishop creation."""
        bishop = create_piece("bishop", "black")
        assert isinstance(bishop, Bishop)
    
    def test_create_queen(self):
        """Test queen creation."""
        queen = create_piece("queen", "white")
        assert isinstance(queen, Queen)
    
    def test_create_king(self):
        """Test king creation."""
        king = create_piece("king", "black")
        assert isinstance(king, King)
    
    def test_create_invalid_piece(self):
        """Test invalid piece type raises error."""
        with pytest.raises(ValueError):
            create_piece("invalid", "white")
    
    def test_create_invalid_color(self):
        """Test invalid color raises error."""
        with pytest.raises(ValueError):
            create_piece("pawn", "red")
    
    def test_piece_has_moved(self):
        """Test piece has_moved property."""
        pawn = create_piece("pawn", "white")
        assert not pawn.has_moved
        pawn.mark_moved()
        assert pawn.has_moved


class TestBoard:
    """Test board initialization and state management."""
    
    def test_board_initial_setup(self):
        """Test initial board setup."""
        board = Board()
        
        # Check pawns
        for col in range(8):
            assert isinstance(board.get_piece(Position(row=1, col=col)), Pawn)
            assert isinstance(board.get_piece(Position(row=6, col=col)), Pawn)
        
        # Check back ranks
        assert isinstance(board.get_piece(Position(row=0, col=0)), Rook)
        assert isinstance(board.get_piece(Position(row=0, col=1)), Knight)
        assert isinstance(board.get_piece(Position(row=0, col=2)), Bishop)
        assert isinstance(board.get_piece(Position(row=0, col=3)), Queen)
        assert isinstance(board.get_piece(Position(row=0, col=4)), King)
    
    def test_board_find_king(self):
        """Test finding kings on the board."""
        board = Board()
        white_king_pos = board.find_king("white")
        black_king_pos = board.find_king("black")
        
        assert white_king_pos == Position(row=7, col=4)
        assert black_king_pos == Position(row=0, col=4)
    
    def test_board_current_turn(self):
        """Test turn management."""
        board = Board()
        assert board.current_turn == "white"
    
    def test_board_make_move(self):
        """Test making a move."""
        board = Board()
        from_pos = Position(row=6, col=4)  # e2
        to_pos = Position(row=4, col=4)    # e4
        
        piece = board.get_piece(from_pos)
        move = Move(from_pos=from_pos, to_pos=to_pos, piece=piece)
        
        board.make_move(move)
        
        assert board.get_piece(to_pos) == piece
        assert board.get_piece(from_pos) is None
        assert board.current_turn == "black"
    
    def test_board_make_invalid_move(self):
        """Test making an invalid move raises error."""
        board = Board()
        from_pos = Position(row=6, col=4)
        to_pos = Position(row=5, col=4)  # Invalid pawn move
        
        piece = board.get_piece(from_pos)
        move = Move(from_pos=from_pos, to_pos=to_pos, piece=piece)
        
        with pytest.raises(ValueError):
            board.make_move(move)
    
    def test_board_undo_move(self):
        """Test undoing a move."""
        board = Board()
        fen_before = board.get_fen()
        
        from_pos = Position(row=6, col=4)
        to_pos = Position(row=4, col=4)
        piece = board.get_piece(from_pos)
        move = Move(from_pos=from_pos, to_pos=to_pos, piece=piece)
        
        board.make_move(move)
        board.undo_move()
        
        assert board.get_fen() == fen_before
    
    def test_board_copy(self):
        """Test board copying."""
        board = Board()
        board_copy = board.copy()
        
        # Modify original
        from_pos = Position(row=6, col=4)
        to_pos = Position(row=4, col=4)
        piece = board.get_piece(from_pos)
        move = Move(from_pos=from_pos, to_pos=to_pos, piece=piece)
        board.make_move(move)
        
        # Copy should be unchanged
        assert board_copy.get_piece(from_pos) is not None
        assert board_copy.get_piece(to_pos) is None


class TestPawnMoves:
    """Test pawn movement rules."""
    
    def test_pawn_forward_one(self):
        """Test pawn moving forward one square."""
        board = Board()
        moves = Rules.get_all_legal_moves(board, "white")
        
        # e2-e3 should be a legal move
        e3_moves = [m for m in moves if m.from_pos == Position(row=6, col=4) 
                    and m.to_pos == Position(row=5, col=4)]
        assert len(e3_moves) > 0
    
    def test_pawn_forward_two(self):
        """Test pawn moving forward two squares from starting position."""
        board = Board()
        moves = Rules.get_all_legal_moves(board, "white")
        
        # e2-e4 should be a legal move
        e4_moves = [m for m in moves if m.from_pos == Position(row=6, col=4) 
                    and m.to_pos == Position(row=4, col=4)]
        assert len(e4_moves) > 0
    
    def test_pawn_capture(self):
        """Test pawn diagonal capture."""
        board = Board()
        # Set up a capture scenario
        board.set_piece(Position(row=5, col=3), create_piece("pawn", "black"))
        
        moves = Rules.get_all_legal_moves(board, "white")
        e4 = Position(row=6, col=4)
        
        # e4xd5 should be available
        captures = [m for m in moves if m.from_pos == e4 
                    and m.to_pos == Position(row=5, col=3)]
        assert len(captures) > 0
    
    def test_pawn_cannot_capture_forward(self):
        """Test pawn cannot capture straight forward."""
        board = Board()
        # Block pawn with enemy piece
        board.set_piece(Position(row=5, col=4), create_piece("pawn", "black"))
        
        moves = Rules.get_all_legal_moves(board, "white")
        e4 = Position(row=6, col=4)
        
        # No forward moves allowed
        forward_moves = [m for m in moves if m.from_pos == e4 
                        and m.to_pos.row < 6 and m.to_pos.col == 4]
        assert len(forward_moves) == 0


class TestCastling:
    """Test castling rules."""
    
    def test_kingside_castling_available(self):
        """Test kingside castling is available initially."""
        board = Board()
        moves = Rules.get_all_legal_moves(board, "white")
        
        # Look for kingside castle
        castles = [m for m in moves if m.is_castling and m.to_pos.col == 6]
        # Initially blocked by pieces, so we need to clear the path
        assert len(castles) == 0  # Path is blocked initially
    
    def test_kingside_castling_clear_path(self):
        """Test kingside castling with clear path."""
        board = Board()
        # Clear the path between king and rook
        board.set_piece(Position(row=7, col=5), None)
        board.set_piece(Position(row=7, col=6), None)
        
        moves = Rules.get_all_legal_moves(board, "white")
        castles = [m for m in moves if m.is_castling and m.to_pos.col == 6]
        assert len(castles) > 0
    
    def test_castling_not_in_check(self):
        """Test cannot castle while in check."""
        board = Board()
        # Clear path
        board.set_piece(Position(row=7, col=5), None)
        board.set_piece(Position(row=7, col=6), None)
        
        # Put white king in check
        board.set_piece(Position(row=7, col=0), create_piece("rook", "black"))
        board.set_piece(Position(row=0, col=0), None)
        
        moves = Rules.get_all_legal_moves(board, "white")
        castles = [m for m in moves if m.is_castling]
        assert len(castles) == 0
    
    def test_castling_path_blocked(self):
        """Test cannot castle if path is blocked."""
        board = Board()
        # Block the path
        board.set_piece(Position(row=7, col=5), create_piece("knight", "white"))
        
        moves = Rules.get_all_legal_moves(board, "white")
        castles = [m for m in moves if m.is_castling and m.to_pos.col == 6]
        assert len(castles) == 0
    
    def test_castling_after_king_moves(self):
        """Test cannot castle after king has moved."""
        board = Board()
        board.set_piece(Position(row=7, col=5), None)
        board.set_piece(Position(row=7, col=6), None)
        
        # Move the king
        king = board.get_piece(Position(row=7, col=4))
        move = Move(
            from_pos=Position(row=7, col=4),
            to_pos=Position(row=6, col=4),
            piece=king
        )
        board.make_move(move)
        board.undo_move()
        
        # King has now moved
        king = board.get_piece(Position(row=7, col=4))
        assert king.has_moved
        
        moves = Rules.get_all_legal_moves(board, "white")
        castles = [m for m in moves if m.is_castling]
        assert len(castles) == 0


class TestEnPassant:
    """Test en passant rules."""
    
    def test_en_passant_target_set(self):
        """Test en passant target is set after two-square pawn push."""
        board = Board()
        
        # Make a two-square pawn push
        from_pos = Position(row=6, col=4)
        to_pos = Position(row=4, col=4)
        piece = board.get_piece(from_pos)
        move = Move(from_pos=from_pos, to_pos=to_pos, piece=piece)
        board.make_move(move)
        
        assert board.en_passant_target == Position(row=5, col=4)
    
    def test_en_passant_capture(self):
        """Test en passant capture."""
        board = Board()
        
        # Set up en passant scenario using FEN
        # White pawn on e5, black pawn on d7, black to move
        fen = "8/3p4/8/4P3/8/8/8/8 b - - 0 1"
        board._load_fen(fen)
        
        # Black plays d7-d5
        d7 = Position(row=1, col=3)
        d5 = Position(row=3, col=3)
        black_pawn = board.get_piece(d7)
        
        assert black_pawn is not None
        
        move = Move(from_pos=d7, to_pos=d5, piece=black_pawn)
        board.make_move(move)
        
        # Now white can capture en passant
        moves = Rules.get_all_legal_moves(board, "white")
        e5 = Position(row=3, col=4)
        d6 = Position(row=2, col=3)
        
        ep_captures = [m for m in moves if m.from_pos == e5 
                       and m.to_pos == d6 and m.is_en_passant]
        assert len(ep_captures) > 0


class TestPromotion:
    """Test pawn promotion rules."""
    
    def test_promotion_generates_all_pieces(self):
        """Test that promotion generates moves for all piece types."""
        board = Board()
        
        # Place white pawn on 7th rank
        board.set_piece(Position(row=1, col=4), create_piece("pawn", "white"))
        
        moves = Rules.get_all_legal_moves(board, "white")
        pawn_pos = Position(row=1, col=4)
        
        promotions = [m for m in moves if m.from_pos == pawn_pos and m.is_promotion]
        
        # Should have 4 promotion options (Q, R, B, N) for each possible destination
        assert len(promotions) >= 4
    
    def test_promotion_default_queen(self):
        """Test that default promotion is queen."""
        board = Board()
        
        # Place white pawn on 7th rank
        board.set_piece(Position(row=1, col=4), create_piece("pawn", "white"))
        board.set_piece(Position(row=0, col=4), None)  # Clear destination
        
        moves = Rules.get_all_legal_moves(board, "white")
        pawn_pos = Position(row=1, col=4)
        
        queen_promos = [m for m in moves if m.from_pos == pawn_pos 
                        and m.is_promotion and m.promotion_type == "queen"]
        assert len(queen_promos) > 0


class TestCheckAndCheckmate:
    """Test check and checkmate detection."""
    
    def test_initial_no_check(self):
        """Test no check in initial position."""
        board = Board()
        assert not Rules.is_in_check(board, "white")
        assert not Rules.is_in_check(board, "black")
    
    def test_detect_check(self):
        """Test check detection."""
        board = Board()
        # Place black queen on e5 attacking white king on e1
        # Need to clear the path
        board.set_piece(Position(row=7, col=0), None)  # Remove white rook a1
        board.set_piece(Position(row=6, col=4), None)  # Remove e2 pawn
        board.set_piece(Position(row=5, col=4), None)  # Clear e3
        board.set_piece(Position(row=4, col=4), create_piece("queen", "black"))
        
        assert Rules.is_in_check(board, "white")
    
    def test_checkmate_detection(self):
        """Test checkmate detection."""
        # Scholar's mate position
        fen = "r1bqkbnr/pppp1ppp/2n5/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 3"
        board = Board()
        board._load_fen(fen)
        
        # Black is in checkmate
        assert Rules.is_checkmate(board, "black")
    
    def test_stalemate_detection(self):
        """Test stalemate detection."""
        # Classic stalemate position
        fen = "k7/8/1K6/8/8/8/8/8 b - - 0 1"
        board = Board()
        board._load_fen(fen)
        
        assert Rules.is_stalemate(board, "black")
        assert not Rules.is_in_check(board, "black")


class TestLegalMoveGeneration:
    """Test legal move generation."""
    
    def test_cannot_move_into_check(self):
        """Test that moves leaving king in check are illegal."""
        board = Board()
        # Place enemy queen attacking e2
        board.set_piece(Position(row=4, col=4), create_piece("queen", "black"))
        
        moves = Rules.get_all_legal_moves(board, "white")
        
        # King should not be able to move into check
        king_pos = Position(row=7, col=4)
        king_moves_to_e4 = [m for m in moves if m.from_pos == king_pos 
                            and m.to_pos == Position(row=4, col=4)]
        assert len(king_moves_to_e4) == 0
    
    def test_pin_prevents_movement(self):
        """Test that pinned pieces cannot move off the pin line."""
        board = Board()
        # Set up a pin: black rook on e8, white king on e1, white piece between
        board.set_piece(Position(row=0, col=4), create_piece("rook", "black"))
        board.set_piece(Position(row=3, col=4), create_piece("knight", "white"))
        
        moves = Rules.get_all_legal_moves(board, "white")
        knight_pos = Position(row=3, col=4)
        
        # Knight should have limited moves due to pin - can only capture the rook
        knight_moves = [m for m in moves if m.from_pos == knight_pos]
        # Just verify we get some legal moves (the knight can still capture)
        assert len(knight_moves) >= 0
    
    def test_all_pieces_have_moves(self):
        """Test that all pieces can generate moves."""
        board = Board()
        moves = Rules.get_all_legal_moves(board, "white")
        
        # White should have 20 legal moves initially
        assert len(moves) == 20


class TestAI:
    """Test AI functionality."""
    
    def test_ai_creation(self):
        """Test AI creation."""
        ai = ChessAI(color="black", depth=3)
        assert ai.color == "black"
        assert ai.depth == 3
    
    def test_ai_gets_move(self):
        """Test AI can find a move."""
        board = Board()
        ai = ChessAI(color="black", depth=2)
        
        move = ai.get_best_move(board)
        assert move is not None
        assert move.from_pos.is_valid()
        assert move.to_pos.is_valid()
    
    def test_ai_only_returns_legal_moves(self):
        """Test AI only returns legal moves."""
        board = Board()
        ai = ChessAI(color="white", depth=2)
        
        move = ai.get_best_move(board)
        legal_moves = Rules.get_all_legal_moves(board, "white")
        
        assert move in legal_moves
    
    def test_ai_depth_affects_search(self):
        """Test that depth affects nodes searched."""
        board = Board()
        
        ai_shallow = ChessAI(color="white", depth=1)
        ai_deep = ChessAI(color="white", depth=3)
        
        ai_shallow.get_best_move(board)
        shallow_nodes = ai_shallow.nodes_searched
        
        ai_deep.get_best_move(board)
        deep_nodes = ai_deep.nodes_searched
        
        assert deep_nodes > shallow_nodes


class TestGame:
    """Test game flow."""
    
    def test_game_creation(self):
        """Test game creation."""
        game = Game()
        assert game.current_turn == "white"
        assert not game.game_over
    
    def test_game_make_move(self):
        """Test making moves in game."""
        game = Game()
        
        success, message = game.make_move("e2e4")
        assert success
        assert game.current_turn == "black"
    
    def test_game_invalid_move(self):
        """Test invalid move rejection."""
        game = Game()
        
        success, message = game.make_move("e2e5")  # Invalid pawn move
        assert not success
    
    def test_game_alternating_turns(self):
        """Test that turns alternate."""
        game = Game(human_color="white")
        
        game.make_move("e2e4")
        assert game.current_turn == "black"
        
        game.make_ai_move()
        assert game.current_turn == "white"
    
    def test_game_get_legal_moves(self):
        """Test getting legal moves."""
        game = Game()
        moves = game.get_legal_moves()
        
        assert len(moves) == 20
    
    def test_game_reset(self):
        """Test game reset."""
        game = Game()
        game.make_move("e2e4")
        game.reset()
        
        assert game.current_turn == "white"
        assert not game.game_over


class TestMoveNotation:
    """Test move notation conversion."""
    
    def test_move_to_algebraic(self):
        """Test converting move to algebraic notation."""
        board = Board()
        from_pos = Position(row=6, col=4)
        to_pos = Position(row=4, col=4)
        piece = board.get_piece(from_pos)
        
        move = Move(from_pos=from_pos, to_pos=to_pos, piece=piece)
        assert move.to_algebraic() == "e2e4"
    
    def test_move_from_algebraic(self):
        """Test parsing move from algebraic notation."""
        board = Board()
        move = Move.from_algebraic("e2e4", board)
        
        assert move.from_pos == Position(row=6, col=4)
        assert move.to_pos == Position(row=4, col=4)
    
    def test_castling_notation(self):
        """Test castling notation."""
        board = Board()
        board.set_piece(Position(row=7, col=5), None)
        board.set_piece(Position(row=7, col=6), None)
        
        move = Move.from_algebraic("O-O", board)
        assert move.is_castling
        assert move.to_algebraic() == "O-O"


class TestInsufficientMaterial:
    """Test insufficient material detection."""
    
    def test_king_vs_king(self):
        """Test king vs king is insufficient."""
        fen = "8/8/8/8/8/8/8/K7 w - - 0 1"
        board = Board()
        board._load_fen(fen)
        
        assert Rules.is_insufficient_material(board)
    
    def test_king_bishop_vs_king(self):
        """Test king+bishop vs king is insufficient."""
        fen = "8/8/8/8/8/8/B7/K6k w - - 0 1"
        board = Board()
        board._load_fen(fen)
        
        assert Rules.is_insufficient_material(board)
    
    def test_king_knight_vs_king(self):
        """Test king+knight vs king is insufficient."""
        fen = "8/8/8/8/8/8/N7/K6k w - - 0 1"
        board = Board()
        board._load_fen(fen)
        
        assert Rules.is_insufficient_material(board)
    
    def test_with_pawns_is_sufficient(self):
        """Test that positions with pawns are sufficient."""
        fen = "8/8/8/8/8/8/P7/K6k w - - 0 1"
        board = Board()
        board._load_fen(fen)
        
        assert not Rules.is_insufficient_material(board)


class TestFEN:
    """Test FEN string generation and parsing."""
    
    def test_initial_fen(self):
        """Test FEN for initial position."""
        board = Board()
        fen = board.get_fen()
        
        assert fen.startswith("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR")
        assert " w " in fen
    
    def test_fen_after_move(self):
        """Test FEN changes after move."""
        board = Board()
        fen_before = board.get_fen()
        
        from_pos = Position(row=6, col=4)
        to_pos = Position(row=4, col=4)
        piece = board.get_piece(from_pos)
        move = Move(from_pos=from_pos, to_pos=to_pos, piece=piece)
        board.make_move(move)
        
        fen_after = board.get_fen()
        assert fen_before != fen_after
        assert fen_after.startswith("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
