"""
文字書きゲーム
a-zの文字を薄い灰色で表示し、マウスドラッグで同じ形を描くゲーム
"""

import argparse
import random
import string

import numpy as np
import pygame
from PIL import Image, ImageDraw, ImageFont


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["sequential", "random"], default="random")
    parser.add_argument("--show_done_button", action="store_true")
    return parser.parse_args()


class LetterTracingGame:
    def __init__(self, sequential, show_done_button):
        pygame.init()

        # 固定値
        self.width = 192
        self.height = 192
        self.font_size = 150

        # Doneボタン表示フラグ
        self.show_done_button = show_done_button

        # 文字表示モード
        self.sequential = sequential
        self.current_index = 0  # 順番モード用

        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Letter Tracing Game")

        self.clock = pygame.time.Clock()

        # 色定義
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.GRAY = (200, 200, 200)
        self.BUTTON_COLOR = (100, 100, 200)
        self.BUTTON_HOVER = (150, 150, 255)
        self.SCORE_BG = (255, 255, 200)  # 薄い黄色（スコア背景）
        self.SCORE_BORDER = (200, 150, 0)  # オレンジ（スコア枠）

        # ゲーム状態
        self.current_letter = None
        self.letter_surface = None
        self.user_surface = None
        self.drawing = False
        self.last_pos = None

        # スコア表示
        self.show_score = False
        self.score = 0.0
        self.score_timer = 0

        # 制限時間（ミリ秒）
        self.time_limit = 5000
        self.start_time = 0

        # Doneボタン
        button_width = 50
        button_height = 20
        self.button_rect = pygame.Rect(
            self.width - button_width - 5,
            self.height - button_height - 5,
            button_width,
            button_height,
        )

        # 文字位置のオフセット
        self.letter_offset_x = 0
        self.letter_offset_y = 0

        # 新しい文字を生成
        self.new_letter()

    def new_letter(self):
        """新しい文字を生成（順番 or ランダム）"""
        if self.sequential:
            # 順番モード: a -> b -> c -> ... -> z -> a -> ...
            self.current_letter = string.ascii_lowercase[self.current_index]
            self.current_index = (self.current_index + 1) % 26
        else:
            # ランダムモード
            self.current_letter = random.choice(string.ascii_lowercase)

        # ランダムなオフセット（-5 ~ +5ピクセル）
        self.letter_offset_x = random.randint(-5, 5)
        self.letter_offset_y = random.randint(-5, 5)

        # 文字のSurfaceを生成
        self.letter_surface = self._create_letter_surface(self.current_letter)

        # ユーザー描画用のSurfaceを初期化
        self.user_surface = pygame.Surface((self.width, self.height))
        self.user_surface.fill(self.WHITE)
        self.user_surface.set_colorkey(self.WHITE)

        # 状態をリセット
        self.drawing = False
        self.last_pos = None
        self.show_score = False
        self.score = 0.0
        self.score_timer = 0
        self.start_time = pygame.time.get_ticks()

    def _create_letter_surface(self, letter):
        """PILを使って文字のSurfaceを生成"""
        # PILで文字を描画
        pil_size = (self.width, self.height)
        pil_image = Image.new("RGB", pil_size, (255, 255, 255))
        draw = ImageDraw.Draw(pil_image)

        # フォントを取得（システムフォント）
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", self.font_size
            )
        except:
            font = ImageFont.load_default()

        # 文字のバウンディングボックスを取得
        bbox = draw.textbbox((0, 0), letter, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # 中央に配置（オフセット付き）
        x = (self.width - text_width) // 2 + self.letter_offset_x
        y = (self.height - text_height) // 2 + self.letter_offset_y

        # 灰色で文字を描画
        draw.text((x, y), letter, fill=(200, 200, 200), font=font)

        # PILイメージをPygame Surfaceに変換
        mode = pil_image.mode
        size = pil_image.size
        data = pil_image.tobytes()

        py_image = pygame.image.fromstring(data, size, mode)

        return py_image

    def calculate_iou(self):
        """Intersection over Unionを計算"""
        # Surfaceをnumpy配列に変換
        letter_array = pygame.surfarray.array3d(self.letter_surface)
        user_array = pygame.surfarray.array3d(self.user_surface)

        # グレースケール化（簡易版: 黒くないピクセルを1とする）
        # 文字Surface: 灰色(200,200,200)以外を0
        letter_mask = np.any(letter_array != [255, 255, 255], axis=2).astype(np.float32)

        # ユーザーSurface: 白(255,255,255)以外を1
        user_mask = np.any(user_array != [255, 255, 255], axis=2).astype(np.float32)

        # Intersection and Union
        intersection = np.sum(letter_mask * user_mask)
        union = np.sum(np.maximum(letter_mask, user_mask))

        if union == 0:
            return 0.0

        iou = intersection / union
        return iou

    def handle_events(self):
        """イベント処理"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            # スコア表示中はマウス操作を無効化
            if self.show_score:
                continue

            # マウスボタンが押された
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # 左クリック
                    # Doneボタンのクリック判定
                    if self.show_done_button and self.button_rect.collidepoint(event.pos):
                        self._on_done_clicked()
                    else:
                        self.drawing = True
                        self.last_pos = event.pos
                        # 最初のクリック位置にも円を描画
                        pygame.draw.circle(self.user_surface, self.BLACK, event.pos, 12)

            # マウスボタンが離された
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.drawing = False
                    self.last_pos = None

            # マウス移動
            elif event.type == pygame.MOUSEMOTION:
                if self.drawing:
                    current_pos = event.pos
                    # 円を描画
                    pygame.draw.circle(self.user_surface, self.BLACK, current_pos, 12)
                    self.last_pos = current_pos

        return True

    def _on_done_clicked(self):
        """Doneボタンがクリックされたときの処理"""
        # IoUスコアを計算
        self.score = self.calculate_iou()

        # スコア表示モードに移行
        self.show_score = True
        self.score_timer = pygame.time.get_ticks()

    def update(self):
        """ゲーム状態の更新"""
        # スコア表示中の場合
        if self.show_score:
            # 1秒経過したら次の文字へ
            if pygame.time.get_ticks() - self.score_timer > 1000:
                self.new_letter()
        else:
            # 制限時間チェック
            if pygame.time.get_ticks() - self.start_time > self.time_limit:
                self._on_done_clicked()

    def draw(self):
        """画面描画"""
        # 背景を白で塗りつぶし
        self.screen.fill(self.WHITE)

        if self.show_score:
            # スコア表示
            self._draw_score()
        else:
            # 文字を描画
            self.screen.blit(self.letter_surface, (0, 0))

            # ユーザーの描画を表示
            self.screen.blit(self.user_surface, (0, 0))

            # Doneボタンを描画
            if self.show_done_button:
                self._draw_button()

        pygame.display.flip()

    def _draw_button(self):
        """Doneボタンを描画"""
        # マウスホバー判定
        mouse_pos = pygame.mouse.get_pos()
        color = self.BUTTON_HOVER if self.button_rect.collidepoint(mouse_pos) else self.BUTTON_COLOR

        pygame.draw.rect(self.screen, color, self.button_rect, border_radius=2)

        # テキスト
        font = pygame.font.Font(None, 16)
        text = font.render("Done", True, self.WHITE)
        text_rect = text.get_rect(center=self.button_rect.center)
        self.screen.blit(text, text_rect)

    def _draw_score(self):
        """スコアを画面中央に表示（OCR検出しやすい形式）"""
        # スコア表示用の矩形（黄色い背景）
        score_box_width = 150
        score_box_height = 80
        score_box_x = (self.width - score_box_width) // 2
        score_box_y = (self.height - score_box_height) // 2

        score_box_rect = pygame.Rect(score_box_x, score_box_y, score_box_width, score_box_height)

        # 背景（薄い黄色）
        pygame.draw.rect(self.screen, self.SCORE_BG, score_box_rect)

        # 枠線（オレンジ）
        pygame.draw.rect(self.screen, self.SCORE_BORDER, score_box_rect, 2)

        # "Score:" ラベル
        label_font = pygame.font.Font(None, 24)
        label_text = label_font.render("Score:", True, self.BLACK)
        label_rect = label_text.get_rect(center=(self.width // 2, self.height // 2 - 15))
        self.screen.blit(label_text, label_rect)

        # スコア値
        score_font = pygame.font.Font(None, 36)
        score_text = f"{self.score:.2f}"
        score_text_surface = score_font.render(score_text, True, self.BLACK)
        score_text_rect = score_text_surface.get_rect(
            center=(self.width // 2, self.height // 2 + 15)
        )
        self.screen.blit(score_text_surface, score_text_rect)

    def get_screen_array(self):
        """画面をnumpy配列として取得（Gymnasium環境用）"""
        # Pygameの画面をnumpy配列に変換
        array = pygame.surfarray.array3d(self.screen)
        # (width, height, 3) -> (height, width, 3)に転置
        array = np.transpose(array, (1, 0, 2))
        return array

    def run(self):
        """ゲームループ"""
        running = True
        while running:
            running = self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(60)

        pygame.quit()


if __name__ == "__main__":
    args = parse_args()

    sequential = args.mode == "sequential"
    game = LetterTracingGame(sequential, args.show_done_button)
    game.run()
