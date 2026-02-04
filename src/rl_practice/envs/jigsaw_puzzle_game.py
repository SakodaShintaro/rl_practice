# SPDX-License-Identifier: MIT
import argparse
import glob
import random
import tkinter as tk

from PIL import Image, ImageChops, ImageDraw, ImageTk


def parse_args():
    parser = argparse.ArgumentParser(description="Jigsaw Puzzle Game")
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--grid_size", type=int, required=True)
    return parser.parse_args()


def generate_edges(grid_size):
    h_edges = []
    for row in range(grid_size - 1):
        row_edges = []
        for col in range(grid_size):
            row_edges.append(random.choice([-1, 1]))
        h_edges.append(row_edges)

    v_edges = []
    for row in range(grid_size):
        row_edges = []
        for col in range(grid_size - 1):
            row_edges.append(random.choice([-1, 1]))
        v_edges.append(row_edges)

    return h_edges, v_edges


def create_piece_mask(piece_size, tab_size, top, right, bottom, left):
    size = piece_size + tab_size * 2
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)

    offset = tab_size
    draw.rectangle([offset, offset, offset + piece_size, offset + piece_size], fill=255)

    cx = offset + piece_size // 2
    if top == 1:
        draw.ellipse(
            [cx - tab_size, 0, cx + tab_size, tab_size * 2],
            fill=255,
        )
    elif top == -1:
        draw.ellipse(
            [cx - tab_size, 0, cx + tab_size, tab_size * 2],
            fill=0,
        )

    if bottom == 1:
        draw.ellipse(
            [
                cx - tab_size,
                offset + piece_size - tab_size,
                cx + tab_size,
                offset + piece_size + tab_size,
            ],
            fill=255,
        )
    elif bottom == -1:
        draw.ellipse(
            [
                cx - tab_size,
                offset + piece_size - tab_size,
                cx + tab_size,
                offset + piece_size + tab_size,
            ],
            fill=0,
        )

    cy = offset + piece_size // 2
    if left == 1:
        draw.ellipse(
            [0, cy - tab_size, tab_size * 2, cy + tab_size],
            fill=255,
        )
    elif left == -1:
        draw.ellipse(
            [0, cy - tab_size, tab_size * 2, cy + tab_size],
            fill=0,
        )

    if right == 1:
        draw.ellipse(
            [
                offset + piece_size - tab_size,
                cy - tab_size,
                offset + piece_size + tab_size,
                cy + tab_size,
            ],
            fill=255,
        )
    elif right == -1:
        draw.ellipse(
            [
                offset + piece_size - tab_size,
                cy - tab_size,
                offset + piece_size + tab_size,
                cy + tab_size,
            ],
            fill=0,
        )

    return mask


def create_pieces(image, grid_size, h_edges, v_edges):
    piece_size = image.size[0] // grid_size
    tab_size = piece_size // 4
    ext_size = piece_size + tab_size * 2
    pieces = []

    extended = Image.new(
        "RGB", (image.size[0] + tab_size * 2, image.size[1] + tab_size * 2), (128, 128, 128)
    )
    rgb_image = image.convert("RGB")
    extended.paste(rgb_image, (tab_size, tab_size))

    for row in range(grid_size):
        for col in range(grid_size):
            top = -h_edges[row - 1][col] if row > 0 else 0
            bottom = h_edges[row][col] if row < grid_size - 1 else 0
            left = -v_edges[row][col - 1] if col > 0 else 0
            right = v_edges[row][col] if col < grid_size - 1 else 0

            mask = create_piece_mask(piece_size, tab_size, top, right, bottom, left)

            left_px = col * piece_size
            top_px = row * piece_size
            crop_region = extended.crop((left_px, top_px, left_px + ext_size, top_px + ext_size))

            piece = Image.new("RGBA", (ext_size, ext_size), (0, 0, 0, 0))
            piece.paste(crop_region, (0, 0), mask=mask)

            edge_mask = mask.point(lambda p: 255 if p > 128 else 0)
            dilated = edge_mask.copy()
            for _ in range(2):
                temp = Image.new("L", (ext_size, ext_size), 0)
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    shifted = Image.new("L", (ext_size, ext_size), 0)
                    shifted.paste(dilated, (dx, dy))
                    temp = Image.composite(dilated, temp, shifted)
                    temp = Image.composite(shifted, temp, shifted)
                dilated = temp

            outline = ImageChops.subtract(dilated, edge_mask)
            outline_rgba = Image.new("RGBA", (ext_size, ext_size), (0, 0, 0, 0))
            outline_rgba.paste((40, 40, 40, 255), mask=outline)
            piece = Image.alpha_composite(piece, outline_rgba)

            pieces.append(piece)

    return pieces, tab_size


class JigsawPuzzle:
    def __init__(self, root, images, grid_size):
        self.root = root
        self.grid_size = grid_size
        self.images = images
        self.dragging = None
        self.drag_offset = (0, 0)

        image_path = random.choice(images)
        image = Image.open(image_path).convert("RGBA")
        image = image.resize((384, 384), Image.NEAREST)
        self.image_size = 384
        self.piece_size = self.image_size // grid_size

        self.h_edges, self.v_edges = generate_edges(grid_size)
        pieces, self.tab_size = create_pieces(image, grid_size, self.h_edges, self.v_edges)
        self.ext_size = self.piece_size + self.tab_size * 2

        self.piece_images = [ImageTk.PhotoImage(p) for p in pieces]
        num_pieces = grid_size * grid_size

        margin = 20
        board_size = self.image_size + self.tab_size * 2 + margin * 2
        self.scatter_width = board_size
        canvas_width = board_size + self.scatter_width + margin
        self.canvas_height = board_size

        self.canvas = tk.Canvas(root, width=canvas_width, height=self.canvas_height, bg="white")
        self.canvas.pack()

        self.board_x = margin + self.tab_size
        self.board_y = margin + self.tab_size
        self.scatter_x = board_size + margin

        self.grid_slots = [None] * num_pieces
        self.piece_positions = []

        for i in range(num_pieces):
            x = self.scatter_x + random.randint(10, self.scatter_width - self.ext_size - 10)
            y = random.randint(10, self.canvas_height - self.ext_size - 10)
            self.piece_positions.append([i, x, y, None])

        self.canvas.bind("<Button-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

        self.draw_all()

        button_frame = tk.Frame(root)
        button_frame.pack()
        tk.Button(button_frame, text="New Puzzle", command=self.new_puzzle).pack(side=tk.LEFT)
        tk.Button(button_frame, text="Show Answer", command=self.show_answer).pack(side=tk.LEFT)

    def draw_all(self):
        self.canvas.delete("all")

        x0 = self.board_x
        y0 = self.board_y
        x1 = self.board_x + self.image_size
        y1 = self.board_y + self.image_size

        self.canvas.create_rectangle(x0, y0, x1, y1, outline="gray", width=2)

        for i in range(1, self.grid_size):
            x = self.board_x + i * self.piece_size
            self.canvas.create_line(x, y0, x, y1, fill="gray", width=1, dash=(4, 4))

        for i in range(1, self.grid_size):
            y = self.board_y + i * self.piece_size
            self.canvas.create_line(x0, y, x1, y, fill="gray", width=1, dash=(4, 4))

        for piece_idx, x, y, slot in self.piece_positions:
            if slot is not None:
                row = slot // self.grid_size
                col = slot % self.grid_size
                draw_x = self.board_x + col * self.piece_size - self.tab_size
                draw_y = self.board_y + row * self.piece_size - self.tab_size
            else:
                draw_x = x
                draw_y = y
            self.canvas.create_image(
                draw_x, draw_y, anchor=tk.NW, image=self.piece_images[piece_idx]
            )

    def on_press(self, event):
        for idx in range(len(self.piece_positions) - 1, -1, -1):
            _, x, y, slot = self.piece_positions[idx]
            if slot is not None:
                row = slot // self.grid_size
                col = slot % self.grid_size
                px = self.board_x + col * self.piece_size - self.tab_size
                py = self.board_y + row * self.piece_size - self.tab_size
            else:
                px = x
                py = y

            if px <= event.x <= px + self.ext_size and py <= event.y <= py + self.ext_size:
                self.dragging = idx
                self.drag_offset = (event.x - px, event.y - py)

                if slot is not None:
                    self.grid_slots[slot] = None
                    self.piece_positions[idx][3] = None

                self.piece_positions[idx][1] = px
                self.piece_positions[idx][2] = py

                item = self.piece_positions.pop(idx)
                self.piece_positions.append(item)
                self.dragging = len(self.piece_positions) - 1
                break

    def on_drag(self, event):
        if self.dragging is None:
            return

        new_x = event.x - self.drag_offset[0]
        new_y = event.y - self.drag_offset[1]
        self.piece_positions[self.dragging][1] = new_x
        self.piece_positions[self.dragging][2] = new_y
        self.draw_all()

    def on_release(self, _event):
        if self.dragging is None:
            return

        piece_x = self.piece_positions[self.dragging][1]
        piece_y = self.piece_positions[self.dragging][2]
        center_x = piece_x + self.ext_size // 2
        center_y = piece_y + self.ext_size // 2

        placed = False
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                slot_idx = row * self.grid_size + col
                slot_x = self.board_x + col * self.piece_size
                slot_y = self.board_y + row * self.piece_size

                if (
                    slot_x - self.tab_size <= center_x <= slot_x + self.piece_size + self.tab_size
                    and slot_y - self.tab_size
                    <= center_y
                    <= slot_y + self.piece_size + self.tab_size
                ):
                    if self.grid_slots[slot_idx] is None:
                        self.grid_slots[slot_idx] = self.dragging
                        self.piece_positions[self.dragging][3] = slot_idx
                        placed = True
                        break
            if placed:
                break

        self.dragging = None
        self.draw_all()
        self.check_completion()

    def check_completion(self):
        for piece_idx, _x, _y, slot in self.piece_positions:
            if slot != piece_idx:
                return

        self.canvas.create_text(
            self.board_x + self.image_size // 2,
            self.board_y + self.image_size // 2,
            text="Complete!",
            font=("Arial", 32, "bold"),
            fill="green",
        )

    def new_puzzle(self):
        image_path = random.choice(self.images)
        self.load_image(image_path)

    def load_image(self, image_path):
        image = Image.open(image_path).convert("RGBA")
        image = image.resize((384, 384), Image.NEAREST)

        self.h_edges, self.v_edges = generate_edges(self.grid_size)
        pieces, self.tab_size = create_pieces(image, self.grid_size, self.h_edges, self.v_edges)
        self.ext_size = self.piece_size + self.tab_size * 2

        self.piece_images = [ImageTk.PhotoImage(p) for p in pieces]

        num_pieces = self.grid_size * self.grid_size
        self.grid_slots = [None] * num_pieces
        self.piece_positions = []

        for i in range(num_pieces):
            x = self.scatter_x + random.randint(10, self.scatter_width - self.ext_size - 10)
            y = random.randint(10, self.canvas_height - self.ext_size - 10)
            self.piece_positions.append([i, x, y, None])

        self.draw_all()

    def show_answer(self):
        num_pieces = self.grid_size * self.grid_size
        self.grid_slots = list(range(num_pieces))
        self.piece_positions = []

        for i in range(num_pieces):
            self.piece_positions.append([i, 0, 0, i])

        self.draw_all()


def main():
    args = parse_args()
    images = glob.glob(f"{args.image_dir}/*.png") + glob.glob(f"{args.image_dir}/*.jpg")

    root = tk.Tk()
    root.title("Jigsaw Puzzle")
    JigsawPuzzle(root, images, args.grid_size)
    root.mainloop()


if __name__ == "__main__":
    main()
