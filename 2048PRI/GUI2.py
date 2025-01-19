# pygame_2048.py

import sys
import pygame
import numpy as np
import multiprocessing as mp
from math import log2

# Import your classes/functions
import Game2048 as Game  # your Game2048 class
from Expectiminimax import getNextBestMoveExpectiminimax

############################
# Game & UI Settings
############################
boardSize = 4  # Your 4x4 game
width, height = 480, 500  # a little space for score
FPS = 20

# Colors
BLACK       = (0,   0,   0)
WHITE       = (255, 255, 255)
FONT_COLOR  = (82,  52,  42)
BG_COLOR    = (232, 232, 232)

# Regions
PLAY_WIDTH  = 480
PLAY_HEIGHT = 480

############################
# Drawing Helpers
############################
def get_tile_color(value):
    """Generate a color for each tile.
       This is optional and can be changed.
       We'll do a gradient based on log2(value)."""
    if value == 0:
        return BG_COLOR
    # Adjust the 'green' component based on log of tile value
    # (similar idea to your posted code).
    factor = log2(value)
    g_component = 235 - factor*((235 - 52)/(boardSize**2))
    return (235, max(52, g_component), 52)

def draw_board(screen, game, tileFont, scoreFont):
    """
    Draws the entire 4x4 board and the score.
    """
    # Fill background
    screen.fill(BLACK)

    board = game.get_board()
    for i in range(boardSize):
        for j in range(boardSize):
            tile_val = board[i][j]
            color = get_tile_color(tile_val)

            # Each tile is a square in the 480x480 region
            tile_w = PLAY_WIDTH // boardSize
            tile_h = PLAY_HEIGHT // boardSize

            rect_x = j * tile_w
            rect_y = i * tile_h
            rect = pygame.Rect(rect_x, rect_y, tile_w, tile_h)

            # Draw background
            pygame.draw.rect(screen, color, rect)

            # Draw tile border
            pygame.draw.rect(screen, FONT_COLOR, rect, 1)

            # Number text
            if tile_val != 0:
                number_text = str(tile_val)
                fontImage = tileFont.render(number_text, True, FONT_COLOR)

                # Auto-scale if text is too wide
                if fontImage.get_width() > tile_w:
                    ratio = tile_w / fontImage.get_width()
                    new_h = int(fontImage.get_height() * ratio)
                    fontImage = pygame.transform.scale(fontImage, (tile_w, new_h))

                # Center the text in the tile
                text_x = rect_x + (tile_w - fontImage.get_width())//2
                text_y = rect_y + (tile_h - fontImage.get_height())//2
                screen.blit(fontImage, (text_x, text_y))

    # Draw Score below the board
    score_str = f"Score: {game.get_score():,}"
    scoreImage = scoreFont.render(score_str, True, WHITE)
    screen.blit(scoreImage, (5, PLAY_HEIGHT + 5))

############################
# Main Loop
############################
def main():
    # Initialize pygame
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("2048 - Expectiminimax PyGame")
    clock = pygame.time.Clock()

    # Fonts
    tileFont = pygame.font.SysFont("", 72)
    scoreFont = pygame.font.SysFont("", 22)

    # Create game & single shared Pool
    game = Game.Game2048()
    mp.freeze_support()
    mp.set_start_method('spawn', force=True)
    pool = mp.Pool(processes=4)

    # Main loop
    running = True
    while running:
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # Optional manual key controls if you want them:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    game.reset()  # reset the game

                # If you want arrow keys to move manually
                elif event.key == pygame.K_LEFT:
                    game.move_left()
                elif event.key == pygame.K_RIGHT:
                    game.move_right()
                elif event.key == pygame.K_UP:
                    game.move_up()
                elif event.key == pygame.K_DOWN:
                    game.move_down()

        # If the game can still move, let AI pick the best move each frame
        # (Remove or reduce how often you do the AI move if it's too fast).
        if game.can_move():
            best_move = getNextBestMoveExpectiminimax(game, pool, depth=2)
            if best_move is not None:
                if best_move == "left":
                    game.move_left()
                elif best_move == "right":
                    game.move_right()
                elif best_move == "up":
                    game.move_up()
                elif best_move == "down":
                    game.move_down()
            else:
                # No valid move returned => treat as game over
                pass
        else:
            # No more moves, game is over
            pass

        # Draw & update
        draw_board(screen, game, tileFont, scoreFont)
        pygame.display.flip()
        clock.tick(FPS)

    # Clean up
    pool.close()
    pool.terminate()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
