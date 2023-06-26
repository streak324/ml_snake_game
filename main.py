import arcade
import numpy

INITIAL_SCREEN_WIDTH = 1280
INITIAL_SCREEN_HEIGHT = 720
SCREEN_TITLE = "Snake"
PIXELS_PER_BOARD_TILE = 32
BOARD_WIDTH = 16
BOARD_HEIGHT = 16
INITIAL_SNAKE_LENGTH = 3

PLAYER_CONTROLLED = True
PLAYER_SNAKE_STEP_DELAY = 0.1 # seconds

class SnakeGame:
	MOVE_UP = 0
	MOVE_LEFT = 1
	MOVE_RIGHT = 2
	MOVE_DOWN = 3

	OPPOSITE_MOVE_DIRS = (MOVE_DOWN, MOVE_RIGHT, MOVE_LEFT, MOVE_UP)

	TILE_SNAKE = 1
	TILE_FOOD = 2

	def __init__(self, board_width: int, board_height: int):
		self.snake_tentative_move_dir = SnakeGame.MOVE_LEFT
		self.snake_move_dir = SnakeGame.MOVE_LEFT
		self.board = numpy.zeros([board_width, board_height], dtype=numpy.int8)
		self.snake = []
		for i in range(INITIAL_SNAKE_LENGTH):
			self.snake.append((int(board_width - INITIAL_SNAKE_LENGTH - 1 + i), int(board_height/2)))
			self.board[self.snake[i][1]][self.snake[i][0]] = SnakeGame.TILE_SNAKE
	def step(self):
		if SnakeGame.OPPOSITE_MOVE_DIRS[self.snake_tentative_move_dir] != self.snake_move_dir:
			self.snake_move_dir = self.snake_tentative_move_dir
		dx = 0
		dy = 0
		if self.snake_move_dir == SnakeGame.MOVE_UP:
			dy = 1
		elif self.snake_move_dir == SnakeGame.MOVE_LEFT:
			dx = -1
		elif self.snake_move_dir == SnakeGame.MOVE_DOWN:
			dy = -1
		elif self.snake_move_dir == SnakeGame.MOVE_RIGHT:
			dx = 1
		new_pos = (self.snake[0][0]+dx, self.snake[0][1]+dy)

		if new_pos[0] >= len(self.board[0]) or new_pos[0] < 0 or new_pos[1] >= len(self.board[1]) or new_pos[1] < 0:
			# TODO: implement game over
			print("out of bounds")
			return
		if self.board[new_pos[1]][new_pos[0]] == SnakeGame.TILE_SNAKE:
			# TODO: implement game over
			print("snake hit self")
			return
		
		self.board[new_pos[1]][new_pos[0]] = SnakeGame.TILE_SNAKE
		snake_tail = self.snake[len(self.snake)-1]
		self.board[snake_tail[1]][snake_tail[0]] = 0
		for i in reversed(range(len(self.snake)-1)):
			self.snake[i+1] = (self.snake[i][0], self.snake[i][1])
		self.snake[0] = new_pos
		print(self.snake)

	def apply_move_dir(self, move_dir):
		self.snake_tentative_move_dir = move_dir

class MyWindow(arcade.Window):
	def __init__(self):
		self.fps = 0
		super().__init__(INITIAL_SCREEN_WIDTH, INITIAL_SCREEN_HEIGHT, SCREEN_TITLE, resizable=True)
		arcade.set_background_color(arcade.csscolor.CORNFLOWER_BLUE)
		self.snake_game = SnakeGame(BOARD_WIDTH, BOARD_HEIGHT)
		self.time_accum: float = 0
	def setup(self):
		pass

	def on_key_press(self, key: int, modifiers: int):
		if key == arcade.key.UP:
			self.snake_game.apply_move_dir(SnakeGame.MOVE_UP)
		if key == arcade.key.LEFT:
			self.snake_game.apply_move_dir(SnakeGame.MOVE_LEFT)
		if key == arcade.key.DOWN:
			self.snake_game.apply_move_dir(SnakeGame.MOVE_DOWN)
		if key == arcade.key.RIGHT:
			self.snake_game.apply_move_dir(SnakeGame.MOVE_RIGHT)
		pass

	def on_update(self, delta_time):
		self.fps = self.fps * 0.9 + 0.1 * (1.0 / (delta_time + (1 / 16384)))
		self.time_accum += delta_time
		if self.time_accum >= PLAYER_SNAKE_STEP_DELAY:
			self.time_accum = 0
			self.snake_game.step()
		pass

	def on_draw(self):
		self.clear()
		# center the board in the window
		board_x_start = 0
		board_y_start = 0
		board_x_start = (self.width - BOARD_WIDTH * PIXELS_PER_BOARD_TILE) * 0.5
		board_y_start = (self.height - BOARD_HEIGHT * PIXELS_PER_BOARD_TILE) * 0.5
		board_colors = [(170, 213, 80), (158, 204, 69)]
		board_color_idx = 0
		for j in range(BOARD_HEIGHT):
			board_color_idx = (board_color_idx + 1) % len(board_colors)
			for i in range(BOARD_WIDTH):
				x_start = board_x_start + i*PIXELS_PER_BOARD_TILE
				y_start = board_y_start + j*PIXELS_PER_BOARD_TILE
				color = board_colors[board_color_idx]
				if i == self.snake_game.snake[0][0] and j == self.snake_game.snake[0][1]:
					color = (104, 0, 182)
				elif self.snake_game.board[j][i] == SnakeGame.TILE_SNAKE:
					color = (0, 0, 0)
				elif self.snake_game.board[j][i] == SnakeGame.TILE_FOOD:
					color = (233, 30, 54)
				arcade.draw_lrtb_rectangle_filled(x_start, x_start+PIXELS_PER_BOARD_TILE, y_start+PIXELS_PER_BOARD_TILE, y_start, color)
				board_color_idx = (board_color_idx + 1) % len(board_colors)
		arcade.draw_text("FPS: {}".format(self.fps), 0, 0, (0,0,0))

def main():
	window = MyWindow()
	if (INITIAL_SCREEN_WIDTH < BOARD_WIDTH * PIXELS_PER_BOARD_TILE or INITIAL_SCREEN_HEIGHT < BOARD_WIDTH * PIXELS_PER_BOARD_TILE):
		print("WARNING: screen width and screen height should be at least {} and {} respectively to fill the whole board".format(BOARD_WIDTH * PIXELS_PER_BOARD_TILE, BOARD_HEIGHT * PIXELS_PER_BOARD_TILE))
	window.setup()
	arcade.run()

if __name__ == "__main__":
    main()