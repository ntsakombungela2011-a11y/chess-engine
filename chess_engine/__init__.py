"""
Chess Engine - Production-Grade Python Implementation

A fully compliant FIDE chess engine with:
- Complete piece movement rules
- Castling, en passant, and pawn promotion
- Check, checkmate, and stalemate detection
- Minimax AI with alpha-beta pruning
- Clean modular architecture

Architecture:
    pieces.py   - Piece definitions and movement logic
    board.py    - Board representation and state management
    movegen.py  - Legal move generation
    rules.py    - Game rules enforcement (check, checkmate, etc.)
    ai.py       - AI engine with minimax and alpha-beta pruning
    game.py     - Game loop and CLI interface

Usage:
    python -m chess_engine.game

Tests:
    pytest tests/
"""

__version__ = "1.0.0"
__author__ = "Chess Engine Team"
