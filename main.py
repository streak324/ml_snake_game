import arcade

arcade.open_window(800, 600, "Drawing example", True)

arcade.set_background_color(arcade.csscolor.WHITE_SMOKE)

arcade.start_render()
pixels_per_board_cell = 32
board_width = 16
board_height = 16
board_x_start = 0
board_y_start = 0
board_colors = [(170, 213, 80), (158, 204, 69)]
board_color_idx = 0
for i in range(board_width):
	board_color_idx = (board_color_idx + 1) % len(board_colors)
	for j in range(board_height):
		x_start = board_x_start + i*pixels_per_board_cell
		y_start = board_y_start + j*pixels_per_board_cell
		arcade.draw_lrtb_rectangle_filled(x_start, x_start+pixels_per_board_cell, y_start + pixels_per_board_cell, y_start, board_colors[board_color_idx])
		board_color_idx = (board_color_idx + 1) % len(board_colors)

arcade.finish_render()

arcade.run()