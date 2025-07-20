import pygame
import numpy as np
from scipy.ndimage import zoom
from nerual import Model
from categories import getCategories

# Constants
DRAW_GRID_SIZE = 28
INPUT_GRID_SIZE = 28
PIXEL_SIZE = 12
CANVAS_SIZE = DRAW_GRID_SIZE * PIXEL_SIZE
SIDEBAR_WIDTH = 300
HEIGHT = CANVAS_SIZE
WIDTH = CANVAS_SIZE + SIDEBAR_WIDTH

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
LIGHT_GRAY = (230, 230, 230)
BLUE = (0, 100, 255)
GREEN = (0, 200, 0)
RED = (200, 0, 0)

# Setup
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Doodle Classifier")
font = pygame.font.SysFont("Arial", 16)
small_font = pygame.font.SysFont("Arial", 13)

categories = getCategories()
grid = np.zeros((DRAW_GRID_SIZE, DRAW_GRID_SIZE), dtype=np.float32)
model = Model(use_existing=True, folder_path="parameters", add_gaussian_noise=False)
top_predictions = []

def draw_canvas():
    for y in range(DRAW_GRID_SIZE):
        for x in range(DRAW_GRID_SIZE):
            val = int(grid[y][x] * 255)
            pygame.draw.rect(screen, (val, val, val), (x * PIXEL_SIZE, y * PIXEL_SIZE, PIXEL_SIZE, PIXEL_SIZE))

def draw_sidebar():
    pygame.draw.rect(screen, LIGHT_GRAY, (CANVAS_SIZE, 0, SIDEBAR_WIDTH, HEIGHT))

    # Draw "Clear" button
    clear_rect = pygame.Rect(CANVAS_SIZE + 70, HEIGHT - 50, 160, 30)
    pygame.draw.rect(screen, RED, clear_rect)
    screen.blit(font.render("Clear", True, WHITE), (clear_rect.x + 50, clear_rect.y + 5))

    return clear_rect

def draw_predictions_bar():
    bar_x = CANVAS_SIZE + 20
    bar_y = 20
    bar_width = SIDEBAR_WIDTH - 40
    max_bar_height = 18
    spacing = 24

    for i, (idx, prob) in enumerate(top_predictions[:10]):
        bar_length = int(prob * (bar_width - 80))
        label = f"{categories[idx]}"
        screen.blit(small_font.render(label, True, BLACK), (bar_x, bar_y + i * spacing))

        pygame.draw.rect(screen, BLUE, (bar_x + 100, bar_y + i * spacing + 3, bar_length, max_bar_height))
        prob_text = f"{int(prob * 100)}%"
        screen.blit(small_font.render(prob_text, True, BLACK), (bar_x + 100 + bar_length + 5, bar_y + i * spacing + 2))

def update_prediction():
    resized = zoom(grid, (INPUT_GRID_SIZE / DRAW_GRID_SIZE, INPUT_GRID_SIZE / DRAW_GRID_SIZE))
    flat_img = resized.reshape(1, -1) * 255
    probs = model.predict_one(flat_img).output[0]
    return sorted(enumerate(probs), key=lambda x: -x[1])

# Game Loop
running = True
mouse_down = False

while running:
    screen.fill(WHITE)
    draw_canvas()
    clear_btn = draw_sidebar()
    draw_predictions_bar()
    pygame.display.flip()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if clear_btn.collidepoint(event.pos):
                grid[:] = 0
                top_predictions = []

            else:
                mouse_down = True

        elif event.type == pygame.MOUSEBUTTONUP:
            mouse_down = False

    if mouse_down:
        mx, my = pygame.mouse.get_pos()
        if mx < CANVAS_SIZE:
            gx, gy = mx // PIXEL_SIZE, my // PIXEL_SIZE
            if 0 <= gx < DRAW_GRID_SIZE and 0 <= gy < DRAW_GRID_SIZE:
                grid[gy][gx] = 1.0
                top_predictions = update_prediction()

pygame.quit()
