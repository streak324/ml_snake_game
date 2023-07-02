import arcade
import arcade.gui
import numpy
import random
from torch import nn
import torch
import time

INITIAL_SCREEN_WIDTH = 1280
INITIAL_SCREEN_HEIGHT = 720
SCREEN_TITLE = "Snake"
PIXELS_PER_BOARD_TILE = 16
BOARD_WIDTH = 32
BOARD_HEIGHT = 24
INITIAL_SNAKE_LENGTH = 3

USER_INPUT_CONTROLLED = False
USER_SNAKE_STEP_DELAY = 0.1 # seconds
COMPUTER_CONTROLLED = True

NUM_GAMES = 100

class SnakeNeuralNet(nn.Module):
	def __init__(self):
		super().__init__()
		self.flatten = nn.Flatten()
		self.linear_relu_stack = nn.Sequential(
			nn.Linear(2, 4),
			nn.ReLU(),
			nn.Linear(4, 4),
			nn.Softmax(dim=1)
		)
	def forward(self, x):
		x = self.flatten(x)
		logits = self.linear_relu_stack(x)
		return logits

class SnakeGame:
	MOVE_UP = 0
	MOVE_LEFT = 1
	MOVE_DOWN = 2
	MOVE_RIGHT = 3

	OPPOSITE_MOVE_DIRS = (MOVE_DOWN, MOVE_RIGHT, MOVE_UP, MOVE_LEFT)

	TILE_EMPTY = 0
	TILE_SNAKE = 1
	TILE_FOOD = 2

	FOOD_COUNTDOWN_DELAY = 120 # steps since food was last eaten before game is over

	def __init__(self, board_width: int, board_height: int):
		self.snake_tentative_move_dir = SnakeGame.MOVE_LEFT
		self.snake_move_dir = SnakeGame.MOVE_LEFT
		self.board_width = board_width
		self.board_height = board_height
		self.board = numpy.zeros([board_height, board_width], dtype=numpy.int8)
		self.snake = []
		self.steps = 0
		self.is_game_over = False
		for i in range(INITIAL_SNAKE_LENGTH):
			self.snake.append((int(board_width - INITIAL_SNAKE_LENGTH - 1 + i), int(board_height/2)))
			self.board[self.snake[i][1]][self.snake[i][0]] = SnakeGame.TILE_SNAKE
		(self.food_pos, self.food_spawned) = self.spawn_food()

	def spawn_food(self):
		best_score = -1
		found = False
		pos = (0,0)
		for i in range(self.board_height):
			for j in range(self.board_width):
				if (j+1 < self.board_width and self.board[i][j+1] == SnakeGame.TILE_SNAKE) or (j-1 < self.board_width and self.board[i][j-1] == SnakeGame.TILE_SNAKE) or (i+1 < self.board_height and self.board[i+1][j] == SnakeGame.TILE_SNAKE) or (i-1 < self.board_height and self.board[i-1][j] == SnakeGame.TILE_SNAKE):
					continue
				if self.board[i][j] == SnakeGame.TILE_EMPTY:
					score = random.random()

					snake_head = self.snake[0]
					if (self.snake_move_dir == SnakeGame.MOVE_UP and j == self.snake[0] and i > snake_head[1]) or (self.snake_move_dir == SnakeGame.MOVE_LEFT and i == self.snake[1] and j < snake_head[0]) or (self.snake_move_dir == SnakeGame.MOVE_DOWN and j == self.snake[0] and i < snake_head[1]) or (self.snake_move_dir == SnakeGame.MOVE_RIGHT and i == self.snake[1] and j > snake_head[0]):
						score = score * 0.5

					if score > best_score:
						found = True
						pos = (j, i)
						best_score = score
		if found:
			self.board[pos[1]][pos[0]] = SnakeGame.TILE_FOOD
			self.food_pos = pos
		self.food_countdown = SnakeGame.FOOD_COUNTDOWN_DELAY
		return (pos, found)


	def step(self):
		if self.is_game_over:
			return True

		self.food_countdown -= 1
		if self.food_countdown < 0:
			self.food_countdown = 0
			self.is_game_over = True
			return True
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

		if new_pos[0] >= self.board_width or new_pos[0] < 0 or new_pos[1] >= self.board_height or new_pos[1] < 0:
			self.is_game_over = True
			return True
		if self.board[new_pos[1]][new_pos[0]] == SnakeGame.TILE_SNAKE:
			self.is_game_over = True
			return True
		
		prev_tile = self.board[new_pos[1]][new_pos[0]]
		snake_tail = self.snake[len(self.snake)-1]
		self.board[new_pos[1]][new_pos[0]] = SnakeGame.TILE_SNAKE
		self.board[snake_tail[1]][snake_tail[0]] = 0
		for i in reversed(range(len(self.snake)-1)):
			self.snake[i+1] = (self.snake[i][0], self.snake[i][1])
		self.snake[0] = new_pos
		if prev_tile == SnakeGame.TILE_FOOD:
			self.snake.append(snake_tail)
			(self.food_pos, self.food_spawned) = self.spawn_food()
		if self.food_spawned == False:
			self.food_pos = self.snake[0]
		self.steps += 1
		return False

	def apply_move_dir(self, move_dir):
		self.snake_tentative_move_dir = move_dir

class MyWindow(arcade.Window):
	def __init__(self):
		self.fps = 0
		self.device = (
			"cuda"
			if torch.cuda.is_available()
			else "mps"
			if torch.backends.mps.is_available()
			else "cpu"
		)
		print(f"Using {self.device} device")
		super().__init__(INITIAL_SCREEN_WIDTH, INITIAL_SCREEN_HEIGHT, SCREEN_TITLE, resizable=True)
		arcade.set_background_color(arcade.csscolor.CORNFLOWER_BLUE)

		self.one_game_not_over = False
		self.neuralnet = SnakeNeuralNet()
		self.optimizer = torch.optim.SGD(self.neuralnet.parameters(), lr=1e-3)
		self.snake_games = []
		self.game_view_idx = 0
		for i in range(NUM_GAMES):
			self.snake_games.append(SnakeGame(BOARD_WIDTH, BOARD_HEIGHT))
		self.time_accum: float = 0

		self.manager = arcade.gui.UIManager()
		self.manager.enable()
		self.is_retry_button_shown = False
		self.spritelist = arcade.SpriteList(capacity= 2*BOARD_WIDTH*BOARD_HEIGHT)
		self.spritelist.initialize()
		board_colors = [(170, 213, 80), (158, 204, 69)]
		board_color_idx = 0
		board_x_start = (self.width - BOARD_WIDTH * PIXELS_PER_BOARD_TILE) * 0.5
		board_y_start = (self.height - BOARD_HEIGHT * PIXELS_PER_BOARD_TILE) * 0.5
		for i in range(BOARD_HEIGHT):
			board_color_idx = (board_color_idx + 1) % len(board_colors)
			for j in range(BOARD_WIDTH):
				sprite = arcade.SpriteSolidColor(width=PIXELS_PER_BOARD_TILE, height=PIXELS_PER_BOARD_TILE, color = board_colors[board_color_idx])
				sprite.center_x = board_x_start + j*PIXELS_PER_BOARD_TILE+PIXELS_PER_BOARD_TILE/2
				sprite.center_y = board_y_start + i*PIXELS_PER_BOARD_TILE+PIXELS_PER_BOARD_TILE/2
				self.spritelist.append(sprite)
				board_color_idx = (board_color_idx + 1) % len(board_colors)

	def setup(self):
		pass

	def on_key_press(self, key: int, modifiers: int):
		if USER_INPUT_CONTROLLED:
			if key == arcade.key.UP:
				self.snake_games[self.game_view_idx].apply_move_dir(SnakeGame.MOVE_UP)
			if key == arcade.key.LEFT:
				self.snake_games[self.game_view_idx].apply_move_dir(SnakeGame.MOVE_LEFT)
			if key == arcade.key.DOWN:
				self.snake_games[self.game_view_idx].apply_move_dir(SnakeGame.MOVE_DOWN)
			if key == arcade.key.RIGHT:
				self.snake_games[self.game_view_idx].apply_move_dir(SnakeGame.MOVE_RIGHT)
		else:
			if key == arcade.key.LEFT:
				self.game_view_idx = (self.game_view_idx - 1) % len(self.snake_games)
			if key == arcade.key.RIGHT:
				self.game_view_idx = (self.game_view_idx + 1) % len(self.snake_games)
			if key == arcade.key.H:
				view_idx = 0
				best_score = 0
				for i, game in enumerate(self.snake_games):
					if calc_score(game) > best_score:
						view_idx = i
						best_score = calc_score(game)
				print(best_score)
				self.game_view_idx = view_idx
			if key == arcade.key.N:
				j = self.game_view_idx
				for i in  range(len(self.snake_games)-1):
					idx = (j + i) % len(self.snake_games)
					if game.is_game_over == False:
						self.game_view_idx = i
						break

	def on_update(self, delta_time):
		self.fps = self.fps * 0.9 + 0.1 * (1.0 / (delta_time + (1 / 16384)))
		self.time_accum += delta_time
		apply_step = COMPUTER_CONTROLLED
		if apply_step and self.time_accum >= USER_SNAKE_STEP_DELAY:
			self.time_accum = 0
			apply_step = True

		if apply_step:
			start = time.time()
			elapsed_step_time = 0
			if COMPUTER_CONTROLLED:
				dpos_matrix = torch.zeros((len(self.snake_games), 2)).float()
				for i, game in enumerate(self.snake_games):
					dpos_matrix[i] = torch.tensor([[game.food_pos[0] - game.snake[0][0], game.food_pos[1] - game.snake[0][1]]]).float()
				nn_out = self.neuralnet(dpos_matrix)

			#probability distribution function
			pdf = torch.distributions.Categorical(nn_out)
			actions = pdf.sample()
			loss_vector = -pdf.log_prob(actions)
			rewards = torch.zeros(len(self.snake_games))
			self.one_game_not_over = False
			for i, game in enumerate(self.snake_games):
				start_step_time = time.time()
				game.apply_move_dir(actions[i])
				game_over = game.step()
				elapsed_step_time += time.time() - start_step_time
				if game_over:
					if USER_INPUT_CONTROLLED and self.is_retry_button_shown == False:
						restart_button = arcade.gui.UIFlatButton(text="Play Again?", x=self.width/2-100, y=100, width=200, height=50)
						restart_button.on_click = self.on_restart
						self.is_retry_button_shown = True
						self.manager.add(restart_button)
					if COMPUTER_CONTROLLED:
						rewards[i] = 0
				elif COMPUTER_CONTROLLED:
					rewards[i] = calc_score(game)
					self.one_game_not_over = True
			loss = (-pdf.log_prob(actions) * rewards).sum()/len(actions)
			self.optimizer.step()
			self.optimizer.zero_grad()
			print("elapsed time: {}. elapsed step time: {}. loss: {}.".format(time.time() - start, elapsed_step_time, loss))
			if self.one_game_not_over == False:
				print("RESTARTING")
				for i, game in enumerate(self.snake_games):
					self.snake_games[i] = SnakeGame(BOARD_WIDTH, BOARD_HEIGHT)
	
	def on_restart(self, event):
		self.time_accum = 0
		self.snake_games[self.game_view_idx] = SnakeGame(BOARD_WIDTH, BOARD_HEIGHT)
		self.is_retry_button_shown = False
		self.manager.clear()

	def on_resize(self, width, height):
		for i in range(BOARD_HEIGHT):
			for j in range(BOARD_WIDTH):
				sprite = self.spritelist[i*BOARD_WIDTH + j]
				sprite.center_x = (self.width - BOARD_WIDTH * PIXELS_PER_BOARD_TILE) * 0.5 + j*PIXELS_PER_BOARD_TILE+PIXELS_PER_BOARD_TILE/2
				sprite.center_y = (self.height - BOARD_HEIGHT * PIXELS_PER_BOARD_TILE) * 0.5 + i*PIXELS_PER_BOARD_TILE+PIXELS_PER_BOARD_TILE/2
		super().on_resize(width, height)

	def on_draw(self):
		self.clear()
		# center the board in the window
		board_x_start = (self.width - BOARD_WIDTH * PIXELS_PER_BOARD_TILE) * 0.5
		board_y_start = (self.height - BOARD_HEIGHT * PIXELS_PER_BOARD_TILE) * 0.5

		self.spritelist.draw()
		for j in range(BOARD_HEIGHT):
			for i in range(BOARD_WIDTH):
				x_start = board_x_start + i*PIXELS_PER_BOARD_TILE
				y_start = board_y_start + j*PIXELS_PER_BOARD_TILE
				color = (0,0,0)
				if i == self.snake_games[self.game_view_idx].snake[0][0] and j == self.snake_games[self.game_view_idx].snake[0][1]:
					color = (104, 0, 182)
				elif self.snake_games[self.game_view_idx].board[j][i] == SnakeGame.TILE_SNAKE:
					color = (0, 0, 0)
				elif self.snake_games[self.game_view_idx].board[j][i] == SnakeGame.TILE_FOOD:
					color = (233, 30, 54)
				else:
					continue
				arcade.draw_lrtb_rectangle_filled(x_start, x_start+PIXELS_PER_BOARD_TILE, y_start+PIXELS_PER_BOARD_TILE, y_start, color)
		self.manager.draw()
		
		arcade.draw_text("FPS: {}".format(self.fps), start_x=0, start_y=0, color=(0,0,0), font_size=16)
		arcade.draw_text("Score: {}".format(len(self.snake_games[self.game_view_idx].snake)), start_x=0, start_y=self.height-24, color=(0,0,0), font_size=16)
		arcade.draw_text("Hunger Countdown: {}".format(self.snake_games[self.game_view_idx].food_countdown), start_x=0, start_y=self.height-48, color=(0,0,0), font_size=16)
		arcade.draw_text("Time Step: {}".format(self.snake_games[self.game_view_idx].steps), start_x=0, start_y=self.height-72, color=(0,0,0), font_size=16)
		arcade.draw_text("Snake Game: {}".format(self.game_view_idx), start_x=0, start_y=self.height-96, color=(0,0,0), font_size=16)

def calc_score(game: SnakeGame):
	return -(abs(game.food_pos[0] - game.snake[0][0]) + game.food_pos[1] - game.snake[0][1])

def main():
	window = MyWindow()
	window.set_update_rate(1/60)
	if (INITIAL_SCREEN_WIDTH < BOARD_WIDTH * PIXELS_PER_BOARD_TILE or INITIAL_SCREEN_HEIGHT < BOARD_HEIGHT * PIXELS_PER_BOARD_TILE):
		print("WARNING: screen width and screen height should be at least {} and {} respectively to fill the whole board".format(BOARD_WIDTH * PIXELS_PER_BOARD_TILE, BOARD_HEIGHT * PIXELS_PER_BOARD_TILE))
	window.setup()
	arcade.run()

if __name__ == "__main__":
	main()