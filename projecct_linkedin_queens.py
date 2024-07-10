import cv2
import numpy as np
from sklearn.metrics import pairwise_distances_argmin
from mss import mss
import math

def capture_screen_region_opencv_mss(x, y, width, height):
    with mss() as sct:
        monitor = {"top": y, "left": x, "width": width, "height": height}
        screenshot = sct.grab(monitor)
        img = np.array(screenshot)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img

def get_dominant_color(image):
    pixels = np.float32(image.reshape(-1, 3))
    n_colors = 1
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)
    dominant = palette[np.argmax(counts)]
    return dominant.astype(int)

def get_predefined_colors():
    return {
        'orange': [140, 190, 230],
        'blue': [231, 175, 140],
        'green': [147, 198, 159],
        'purple': [183, 153, 211],
        'red': [206, 151, 176],
        'cyan': [100, 122, 234],
        'gray': [204, 204, 203],
        'qing':[205,195,155],
        'pink':[180,150,207],
        'yellow':[130,225,211]
    }

# Capture a single frame
x, y, width, height = 972, 381, 591, 590
img = capture_screen_region_opencv_mss(x, y, width, height)

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Use Canny edge detection
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# Dilate the edges
dilated = cv2.dilate(edges, np.ones((15, 15), np.uint8))

# Apply threshold to get binary image
_, binary = cv2.threshold(dilated, 127, 255, cv2.THRESH_BINARY_INV)

# Find contours
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter contours to find the grid
contour_areas = [cv2.contourArea(c) for c in contours]

# cv2.imshow("con",dilated)
# cv2.imshow("Game", img)

grid_size= int(math.sqrt(len(contour_areas)))
# Grid dimensions
height, width, _ = img.shape
cell_width = width // grid_size
cell_height = height // grid_size


# Predefined color categories (approximate RGB values)
color_categories = get_predefined_colors()

# Convert predefined colors to numpy array for distance calculation
color_values = np.array(list(color_categories.values()))

# Analyze each cell in the grid
dominant_colors = []
for row in range(grid_size):
    row_colors = []
    for col in range(grid_size):
        cell = img[row * cell_height:(row + 1) * cell_height, col * cell_width:(col + 1) * cell_width]
        dominant_color = get_dominant_color(cell)
        row_colors.append(dominant_color)
        # Draw rectangle for visualization
        cv2.rectangle(img, (col*cell_width, row*cell_height), ((col+1)*cell_width, (row+1)*cell_height), (int(dominant_color[0]), int(dominant_color[1]), int(dominant_color[2])), 2)
    dominant_colors.append(row_colors)

# Function to match each dominant color to the nearest predefined color
print(dominant_colors)
def match_color(dominant_colors, color_values):
    labels = []
    for row in dominant_colors:
        row_labels = []
        for color in row:
            closest_color_index = pairwise_distances_argmin([color], color_values)[0]
            row_labels.append(closest_color_index)
        labels.append(row_labels)
    return labels

# Get color labels for each cell
color_labels = match_color(dominant_colors, color_values)

# Print the matched color labels for debugging
color_names = list(color_categories.keys())
# for row in color_labels:
#     print([color_names[label] for label in row])


##################################################################################

color_board = np.array(color_labels)
color_board = color_board.astype(int)
print(color_board)
board_size = len(color_board)

# initialize the board
iBoard = np.full((board_size, board_size), '.', dtype=str)


# check if it is safe to put
def is_safe(board, row, col, color_board):
    # check row and col
    for i in range(board_size):
        if board[row][i] == 'k' or board[i][col] == 'k':
            return False
    # check color
    color = color_board[row][col]
    for r in range(board_size):
        for c in range(board_size):
            if color_board[r][c] == color and board[r][c] == 'k':
                return False
    # check around
    for i in [-1, 1]:
        for j in [-1, 1]:
            if 0 <= row + i < board_size and 0 <= col + j < board_size:
                if board[row + i][col + j] == 'k':
                    return False
    return True

# Recursive functions to solve problems
def solve_n_kings(board, color_board, k_positions, row, col):
    # check if all k been put
    if len(k_positions) == board_size:
        return True

    # if over, go next
    if col >= board_size:
        row += 1
        col = 0

    # if over board size, false
    if row >= board_size:
        return False

    # try put in the current row col.
    if is_safe(board, row, col, color_board):
        board[row][col] = 'k'
        k_positions.append((row, col))
        if solve_n_kings(board, color_board, k_positions, row, col + 1):
            return True
        board[row][col] = '.'
        k_positions.pop()

    # if no, move to next col
    if solve_n_kings(board, color_board, k_positions, row, col + 1):
        return True

    return False









# intialize and solve the problem
k_positions = []
if solve_n_kings(iBoard, color_board, k_positions, 0, 0):
    for row in iBoard:
        print(' '.join(row))
else:
    print("I can't solve")

# draw it in the image
for row in range(board_size):
    for col in range(board_size):
        if iBoard[row][col] == 'k':
            center = (col * cell_width + cell_width // 2-5, row * cell_height + cell_height // 2 + 10)
            cv2.putText(img, 'X', center, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)


cv2.imshow("Problem Solved", img)


# Wait for a key press and close the window
cv2.waitKey(0)
cv2.destroyAllWindows()
