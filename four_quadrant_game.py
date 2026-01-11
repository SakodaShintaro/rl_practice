"""
4分割クリックゲーム
画面を4分割し、1区画だけ赤色、残りは白色
赤色クリック→報酬+1、白色クリック→報酬-1
"""

import argparse
import random

import numpy as np
import pygame


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--time_limit", type=int)
    return parser.parse_args()


class FourQuadrantGame:
    def __init__(self, time_limit):
        pygame.init()

        # 固定値
        self.width = 192
        self.height = 192

        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Four Quadrant Game")

        self.clock = pygame.time.Clock()

        # 色定義
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.SCORE_BG = (255, 255, 200)
        self.SCORE_BORDER = (200, 150, 0)

        # 4分割の矩形を定義
        half_w = self.width // 2
        half_h = self.height // 2
        self.quadrants = [
            pygame.Rect(0, 0, half_w, half_h),  # 左上
            pygame.Rect(half_w, 0, half_w, half_h),  # 右上
            pygame.Rect(0, half_h, half_w, half_h),  # 左下
            pygame.Rect(half_w, half_h, half_w, half_h),  # 右下
        ]

        # スコア表示
        self.show_score = False
        self.score = 0.0
        self.score_timer = 0

        # 制限時間（ミリ秒）
        self.time_limit = time_limit if time_limit is not None else 3000
        self.start_time = 0

        # 現在の正解の区画インデックス
        self.correct_quadrant = 0

        # 新しい問題を生成
        self.new_question()

    def new_question(self):
        """新しい問題を生成"""
        # ランダムに1つの区画を選ぶ
        self.correct_quadrant = random.randint(0, 3)

        # 状態をリセット
        self.show_score = False
        self.score = 0.0
        self.score_timer = 0
        self.start_time = pygame.time.get_ticks()

    def handle_events(self):
        """イベント処理"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

        # マウスボタンの状態を取得
        mouse_pressed = pygame.mouse.get_pressed()[0]

        # スコア表示中でなく、マウスボタンが押されている場合は判定
        if not self.show_score and mouse_pressed:
            pos = pygame.mouse.get_pos()
            self._on_click(pos)

        return True

    def _on_click(self, pos):
        """クリック時の処理"""
        # どの区画がクリックされたか判定
        clicked_quadrant = None
        for i, rect in enumerate(self.quadrants):
            if rect.collidepoint(pos):
                clicked_quadrant = i
                break

        # スコアを計算
        if clicked_quadrant == self.correct_quadrant:
            self.score = 1.0  # 正解
        else:
            self.score = 0.0  # 不正解

        # スコア表示モードに移行
        self.show_score = True
        self.score_timer = pygame.time.get_ticks()

    def update(self):
        """ゲーム状態の更新"""
        # スコア表示中の場合
        if self.show_score:
            # 一定時間経過したら次の問題へ
            if pygame.time.get_ticks() - self.score_timer > 500:
                self.new_question()

    def draw(self):
        """画面描画"""
        # 背景を白で塗りつぶし
        self.screen.fill(self.WHITE)

        if self.show_score:
            # スコア表示
            self._draw_score()
        else:
            # 各区画を描画
            for i, rect in enumerate(self.quadrants):
                if i == self.correct_quadrant:
                    # 正解の区画は赤色
                    pygame.draw.rect(self.screen, self.RED, rect)
                else:
                    # それ以外は白色（すでに背景が白なので枠線だけ描く）
                    pygame.draw.rect(self.screen, self.BLACK, rect, 1)

        pygame.display.flip()

    def _draw_score(self):
        """スコアを画面中央に表示"""
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
        score_text = f"{self.score:.1f}"
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
    game = FourQuadrantGame(args.time_limit)
    game.run()
