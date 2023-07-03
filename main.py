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

NUM_GAMES = 1

class SnakeNeuralNet(nn.Module):
	def __init__(self):
		super().__init__()
		self.flatten = nn.Flatten()
		self.linear_relu_stack = nn.Sequential(
			nn.Linear(6, 8),
			nn.ReLU(),
			nn.Linear(8, 4),
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
		self.ate_food = False
		for i in range(INITIAL_SNAKE_LENGTH):
			self.snake.append((int(board_width - INITIAL_SNAKE_LENGTH - 1 + i), int(board_height/2)))
			self.board[self.snake[i][1]][self.snake[i][0]] = SnakeGame.TILE_SNAKE
		(self.food_pos, self.food_spawned) = self.spawn_food()

	def spawn_food(self):
		best_score = -1
		found = False
		pos = (0,0)
		for i in range(10):
			pos = (random.randrange(0, self.board_width), random.randrange(0, self.board_height))
			if self.board[pos[1]][pos[0]] == SnakeGame.TILE_EMPTY: 
				found = True
				break
		c = pos[0] * self.board_width + pos[1]
		for e in range(self.board_width * self.board_height - 1):
			c = (c + 1) % self.board_width * self.board_height
			if self.board[pos[1]][pos[0]] == SnakeGame.TILE_EMPTY: 
				found = True
				break

		if found:
			self.board[pos[1]][pos[0]] = SnakeGame.TILE_FOOD
			self.food_pos = pos
		self.food_countdown = SnakeGame.FOOD_COUNTDOWN_DELAY
		return (pos, found)


	def step(self):
		if self.is_game_over:
			return True

		self.ate_food = False
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
			self.ate_food = True
			(self.food_pos, self.food_spawned) = self.spawn_food()
		if self.food_spawned == False:
			self.food_pos = self.snake[0]
		self.steps += 1
		return False

	def apply_move_dir(self, move_dir):
		self.snake_tentative_move_dir = move_dir

	def get_obstacle_directions(self):
		left_dist = 0
		right_dist = 0 
		down_dist = 0
		up_dist = 0
		for l in range(1, self.snake[0][0]+1):
			x = self.snake[0][0] - l
			if l < 0:
				break
			tile = self.board[self.snake[0][1]][x]
			if tile != SnakeGame.TILE_EMPTY and tile != SnakeGame.TILE_FOOD:
				break
			left_dist = l
		for x in range(self.snake[0][0]+1, self.board_width):
			tile = self.board[self.snake[0][1]][x]
			if tile != SnakeGame.TILE_EMPTY and tile != SnakeGame.TILE_FOOD:
				break
			right_dist = x - self.snake[0][0]
		for d in range(1, self.snake[0][1]+1):
			y = self.snake[0][1] - d
			if d < 0:
				break
			tile = self.board[y][self.snake[0][0]]
			if tile != SnakeGame.TILE_EMPTY and tile != SnakeGame.TILE_FOOD:
				break
			down_dist = d
		for y in range(self.snake[0][1]+1, self.board_height):
			tile = self.board[y][self.snake[0][0]]
			if tile != SnakeGame.TILE_EMPTY and tile != SnakeGame.TILE_FOOD:
				break
			up_dist = y - self.snake[0][1]
		return (up_dist, left_dist, down_dist, right_dist)

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
		self.max_steps = 0
		self.max_snake_length = 0

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
		if apply_step == False and self.time_accum >= USER_SNAKE_STEP_DELAY:
			self.time_accum = 0
			apply_step = True

		if apply_step:
			start = time.time()
			if COMPUTER_CONTROLLED:
				input_matrix = torch.zeros((len(self.snake_games), 6)).float()
				for i, game in enumerate(self.snake_games):
					inputs = torch.tensor([game.food_pos[0] - game.snake[0][0], game.food_pos[1] - game.snake[0][1], 0, 0, 0, 0]).float()
					obstacle_directions = game.get_obstacle_directions()
					input_move_offset = 2
					for j in range(4):
						inputs[j+input_move_offset] = obstacle_directions[j]
					input_matrix[i] = inputs
				nn_out = self.neuralnet(input_matrix)
				#probability distribution function
				pdf = torch.distributions.Categorical(nn_out)

				explore_actions = pdf.sample()
				actions = explore_actions.clone()
				for i in range(len(actions)):
					# only do do exploring 10% of the time
					if random.random() > 0.1:
						actions[i] = max(nn_out[i])
				rewards = torch.zeros(len(self.snake_games))
				game.apply_move_dir(actions[i])
			for i, game in enumerate(self.snake_games):
				is_game_over = game.step()
				self.max_snake_length = max(self.max_snake_length, len(game.snake))
				self.max_steps = max(game.steps, self.max_steps)
				if COMPUTER_CONTROLLED:
					rewards[i] = calc_score(game)
				if is_game_over:
					if USER_INPUT_CONTROLLED and self.is_retry_button_shown == False:
						restart_button = arcade.gui.UIFlatButton(text="Play Again?", x=self.width/2-100, y=100, width=200, height=50)
						restart_button.on_click = self.on_restart
						self.is_retry_button_shown = True
						self.manager.add(restart_button)
					if COMPUTER_CONTROLLED:
						self.snake_games[i] = SnakeGame(BOARD_WIDTH, BOARD_HEIGHT)
			if COMPUTER_CONTROLLED:
				rewards[i] = calc_score(game)
				loss = (-pdf.log_prob(actions) * rewards).sum()/len(actions)
				self.optimizer.step()
				self.optimizer.zero_grad()
				print("elapsed time: {}.loss: {}.".format(time.time() - start, loss))
	
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
		obstacle_directions = self.snake_games[self.game_view_idx].get_obstacle_directions()
		head_x = self.snake_games[self.game_view_idx].snake[0][0]
		head_y = self.snake_games[self.game_view_idx].snake[0][1]
		obstacle_directions_color = (128, 0, 128, 64)
		for j in range(1,obstacle_directions[SnakeGame.MOVE_UP]+1):
			x = board_x_start + head_x * PIXELS_PER_BOARD_TILE
			y = board_y_start + (head_y + j) * PIXELS_PER_BOARD_TILE
			arcade.draw_lrtb_rectangle_filled(x, x+PIXELS_PER_BOARD_TILE, y+PIXELS_PER_BOARD_TILE, y, obstacle_directions_color)
		for j in range(1,obstacle_directions[SnakeGame.MOVE_LEFT]+1):
			x = board_x_start + (head_x - j) * PIXELS_PER_BOARD_TILE
			y = board_y_start + head_y * PIXELS_PER_BOARD_TILE
			arcade.draw_lrtb_rectangle_filled(x, x+PIXELS_PER_BOARD_TILE, y+PIXELS_PER_BOARD_TILE, y, obstacle_directions_color)
		for j in range(1,obstacle_directions[SnakeGame.MOVE_DOWN]+1):
			x = board_x_start + head_x * PIXELS_PER_BOARD_TILE
			y = board_y_start + (head_y - j) * PIXELS_PER_BOARD_TILE
			arcade.draw_lrtb_rectangle_filled(x, x+PIXELS_PER_BOARD_TILE, y+PIXELS_PER_BOARD_TILE, y, obstacle_directions_color)
		for j in range(1,obstacle_directions[SnakeGame.MOVE_RIGHT]+1):
			x = board_x_start + (head_x + j) * PIXELS_PER_BOARD_TILE
			y = board_y_start + head_y * PIXELS_PER_BOARD_TILE
			arcade.draw_lrtb_rectangle_filled(x, x+PIXELS_PER_BOARD_TILE, y+PIXELS_PER_BOARD_TILE, y, obstacle_directions_color)
		self.manager.draw()
		
		arcade.draw_text("FPS: {}".format(self.fps), start_x=0, start_y=0, color=(0,0,0), font_size=16)
		arcade.draw_text("Score: {}".format(len(self.snake_games[self.game_view_idx].snake)), start_x=0, start_y=self.height-24, color=(0,0,0), font_size=16)
		arcade.draw_text("Hunger Countdown: {}".format(self.snake_games[self.game_view_idx].food_countdown), start_x=0, start_y=self.height-48, color=(0,0,0), font_size=16)
		arcade.draw_text("Time Step: {}".format(self.snake_games[self.game_view_idx].steps), start_x=0, start_y=self.height-72, color=(0,0,0), font_size=16)
		arcade.draw_text("Snake Game: {}".format(self.game_view_idx), start_x=0, start_y=self.height-96, color=(0,0,0), font_size=16)
		arcade.draw_text("Max Steps: {}".format(self.max_steps), start_x=0, start_y=self.height-120, color=(0,0,0), font_size=16)
		arcade.draw_text("Max Snake Length: {}".format(self.max_snake_length), start_x=0, start_y=self.height-144, color=(0,0,0), font_size=16)

def calc_score(game: SnakeGame):
	return 1
	#if game.is_game_over:
	#	return -(game.board_width + game.board_height)
	#current_dist = abs(game.food_pos[0] - game.snake[0][0]) + abs(game.food_pos[1] - game.snake[0][1])
	#prev_dist = abs(game.food_pos[0] - game.snake[1][0]) + abs(game.food_pos[1] - game.snake[1][1])
	#return -(prev_dist - current_dist)
	#current_dist = abs(game.food_pos[0] - game.snake[0][0]) + abs(game.food_pos[1] - game.snake[0][1])
	#prev_dist = abs(game.food_pos[0] - game.snake[1][0]) + abs(game.food_pos[1] - game.snake[1][1])
	#if game.ate_food == True:
	#	return 5
	#elif prev_dist < current_dist:
	#	return 2
	#return 1

def main():
	window = MyWindow()
	window.set_update_rate(1/60)
	if (INITIAL_SCREEN_WIDTH < BOARD_WIDTH * PIXELS_PER_BOARD_TILE or INITIAL_SCREEN_HEIGHT < BOARD_HEIGHT * PIXELS_PER_BOARD_TILE):
		print("WARNING: screen width and screen height should be at least {} and {} respectively to fill the whole board".format(BOARD_WIDTH * PIXELS_PER_BOARD_TILE, BOARD_HEIGHT * PIXELS_PER_BOARD_TILE))
	window.setup()
	arcade.run()

if __name__ == "__main__":
	main()