"""
Chess Game Module

Provides the main game loop and CLI interface for playing chess against the AI.
Also exposes a clean API for external use (e.g., online multiplayer).

Classes:
    Game: Main game controller managing turns, moves, and game state

Usage:
    # Play via CLI
    python -m chess_engine.game
    
    # Use as API
    from chess_engine import Game
    game = Game()
    game.make_move("e2e4")
    print(game.get_legal_moves())
"""

from typing import List, Optional, Tuple

from .pieces import Position, PieceType
from .board import Board, Move
from .rules import Rules
from .ai import ChessAI


class Game:
    """
    Chess game controller.
    
    Manages the game state, handles player input, and coordinates
    between the board, rules, and AI modules.
    
    Attributes:
        board: The current board state
        ai: The AI opponent (if enabled)
        human_color: The color the human player is using
    """
    
    def __init__(self, ai_depth: int = 3, human_color: str = "white"):
        """
        Initialize a new chess game.
        
        Args:
            ai_depth: Search depth for AI (default: 3)
            human_color: Color the human plays ('white' or 'black')
        """
        if human_color not in ("white", "black"):
            raise ValueError(f"Invalid color: {human_color}")
        
        self._board = Board()
        self._ai = ChessAI(color="black" if human_color == "white" else "white", 
                          depth=ai_depth)
        self._human_color = human_color
        self._game_over = False
        self._result = "*"
    
    @property
    def board(self) -> Board:
        """Return the current board state."""
        return self._board
    
    @property
    def ai(self) -> ChessAI:
        """Return the AI instance."""
        return self._ai
    
    @property
    def human_color(self) -> str:
        """Return the human player's color."""
        return self._human_color
    
    @property
    def game_over(self) -> bool:
        """Return whether the game has ended."""
        return self._game_over
    
    @property
    def result(self) -> str:
        """Return the game result."""
        return self._result
    
    @property
    def current_turn(self) -> str:
        """Return whose turn it is."""
        return self._board.current_turn
    
    def get_legal_moves(self) -> List[Move]:
        """
        Get all legal moves for the current player.
        
        Returns:
            List of legal moves
        """
        return Rules.get_all_legal_moves(self._board)
    
    def get_legal_moves_algebraic(self) -> List[str]:
        """
        Get all legal moves in algebraic notation.
        
        Returns:
            List of move strings in algebraic notation
        """
        moves = Rules.get_all_legal_moves(self._board)
        return [move.to_algebraic() for move in moves]
    
    def make_move(self, move_input: str) -> Tuple[bool, str]:
        """
        Make a move specified by the player.
        
        Args:
            move_input: Move in algebraic notation (e.g., 'e2e4', 'O-O')
            
        Returns:
            Tuple of (success, message)
        """
        if self._game_over:
            return False, "Game is over"
        
        if self._board.current_turn != self._human_color:
            return False, f"It's {self._board.current_turn}'s turn"
        
        try:
            move = Move.from_algebraic(move_input, self._board)
        except ValueError as e:
            return False, f"Invalid move format: {e}"
        
        # Validate the move
        legal_moves = Rules.get_all_legal_moves(self._board)
        if move not in legal_moves:
            # Check if the basic move format is valid but just not legal
            piece = self._board.get_piece(move.from_pos)
            if piece is None:
                return False, f"No piece at {move.from_pos.to_algebraic()}"
            if piece.color != self._board.current_turn:
                return False, f"It's {self._board.current_turn}'s turn"
            return False, f"Illegal move: {move_input}"
        
        # Execute the move
        self._board.make_move(move)
        
        # Check game state
        self._update_game_state()
        
        return True, f"Moved {move_input}"
    
    def make_ai_move(self) -> Optional[Move]:
        """
        Let the AI make a move.
        
        Returns:
            The move made by the AI, or None if no moves available
        """
        if self._game_over:
            return None
        
        if self._board.current_turn != self._ai.color:
            return None
        
        move = self._ai.get_best_move(self._board)
        
        if move is None:
            self._update_game_state()
            return None
        
        self._board.make_move(move)
        self._update_game_state()
        
        return move
    
    def _update_game_state(self) -> None:
        """Update the game state after a move."""
        if Rules.is_checkmate(self._board):
            self._game_over = True
            winner = "Black" if self._board.current_turn == "white" else "White"
            self._result = "0-1" if winner == "Black" else "1-0"
        elif Rules.is_stalemate(self._board):
            self._game_over = True
            self._result = "1/2-1/2"
        elif Rules.is_insufficient_material(self._board):
            self._game_over = True
            self._result = "1/2-1/2"
        elif self._board.halfmove_clock >= 100:
            self._game_over = True
            self._result = "1/2-1/2"
    
    def undo_move(self) -> bool:
        """
        Undo the last move.
        
        Returns:
            True if a move was undone, False otherwise
        """
        if self._game_over:
            self._game_over = False
            self._result = "*"
        
        moved = self._board.undo_move()
        return moved is not None
    
    def is_in_check(self) -> bool:
        """Check if the current player's king is in check."""
        return Rules.is_in_check(self._board, self._board.current_turn)
    
    def get_fen(self) -> str:
        """Get the FEN string for the current position."""
        return self._board.get_fen()
    
    def display_board(self) -> None:
        """Print the current board state."""
        print(self._board)
    
    def reset(self) -> None:
        """Reset the game to the starting position."""
        self._board = Board()
        self._game_over = False
        self._result = "*"
    
    def play_cli(self) -> None:
        """
        Run the CLI game loop.
        
        This method provides an interactive command-line interface
        for playing chess against the AI.
        """
        print("=" * 50)
        print("Welcome to Python Chess Engine!")
        print("=" * 50)
        print(f"You are playing as: {self._human_color.upper()}")
        print("Commands:")
        print("  - Enter moves like: e2e4, g1f3, e7e8q (for promotion)")
        print("  - Castling: O-O or O-O-O")
        print("  - 'quit' - Exit the game")
        print("  - 'undo' - Undo last move")
        print("  - 'moves' - Show legal moves")
        print("  - 'flip' - Flip board view")
        print("=" * 50)
        
        flip_view = (self._human_color == "black")
        
        while not self._game_over:
            self.display_board()
            print()
            
            # Show whose turn
            turn_str = f"{self._board.current_turn.upper()}'s turn"
            if self.is_in_check():
                turn_str += " - CHECK!"
            print(turn_str)
            print()
            
            # AI's turn
            if self._board.current_turn == self._ai.color:
                print("AI is thinking...")
                move = self.make_ai_move()
                if move:
                    print(f"AI plays: {move.to_algebraic()}")
                    print(f"Nodes searched: {self._ai.nodes_searched}")
                else:
                    print("AI has no moves")
                continue
            
            # Human's turn
            try:
                user_input = input("Your move: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGame interrupted.")
                break
            
            if user_input.lower() == "quit":
                print("Goodbye!")
                break
            
            if user_input.lower() == "undo":
                if self.undo_move():
                    print("Move undone.")
                    # Undo AI's move too if applicable
                    if self._board.current_turn == self._human_color:
                        self.undo_move()
                        print("AI's move undone.")
                else:
                    print("No moves to undo.")
                continue
            
            if user_input.lower() == "moves":
                moves = self.get_legal_moves_algebraic()
                print(f"Legal moves ({len(moves)}): {', '.join(moves[:20])}{'...' if len(moves) > 20 else ''}")
                continue
            
            if user_input.lower() == "flip":
                flip_view = not flip_view
                print("Board view flipped.")
                continue
            
            # Try to make the move
            success, message = self.make_move(user_input)
            if not success:
                print(f"Invalid: {message}")
            else:
                print(message)
        
        # Game over
        self.display_board()
        print()
        print("=" * 50)
        print("GAME OVER")
        print(f"Result: {self._result}")
        
        if self._result == "1-0":
            print("White wins!")
        elif self._result == "0-1":
            print("Black wins!")
        else:
            print("Draw!")
        print("=" * 50)


def main():
    """Main entry point for CLI game."""
    game = Game(ai_depth=3, human_color="white")
    game.play_cli()


if __name__ == "__main__":
    main()
