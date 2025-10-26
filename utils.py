import cv2
import numpy as np


def concat_images(image_list: list[np.ndarray]) -> np.ndarray:
    """
    image_listを並べて連結する。
    Args:
        image_list (list[np.ndarray]): The list of rgb images to concatenate.
    Returns:
        np.ndarray: The concatenated bgr image (np.uint8).
    """
    max_height = max(img.shape[0] for img in image_list)
    padded_images = []
    for img in image_list:
        padding = np.zeros((max_height - img.shape[0], img.shape[1], 3), dtype=img.dtype)
        padded_img = np.vstack((img, padding))
        padded_images.append(padded_img)
    concat = cv2.hconcat(padded_images)
    bgr_array = cv2.cvtColor(concat, cv2.COLOR_RGB2BGR)
    return bgr_array


def convert_to_uint8(image: np.ndarray) -> np.ndarray:
    """
    画像をuint8形式に変換する。
    Args:
        image (np.ndarray): 元の画像
    Returns:
        np.ndarray: uint8形式の画像
    """
    if image.dtype == np.float32:
        img_converted = image * 255.0
        img_converted = np.clip(img_converted, 0, 255).astype(np.uint8)
        return img_converted
    else:
        return image.copy()


def create_reward_visualization(pred_reward: float, actual_reward: float) -> np.ndarray:
    """
    報酬予測と実際の報酬を可視化する関数（シンプルなテキスト表示）
    """
    height, width = (200, 200)
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # 背景を黒に設定
    img[:, :] = [0, 0, 0]

    # テキストを描画
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2

    # 予測報酬
    pred_text = f"Pred: {pred_reward:.3f}"
    cv2.putText(img, pred_text, (10, 40), font, font_scale, (0, 128, 255), thickness)

    # 実際の報酬
    actual_text = f"Actual: {actual_reward:.3f}"
    cv2.putText(img, actual_text, (10, 80), font, font_scale, (255, 0, 0), thickness)

    # 誤差
    error = abs(pred_reward - actual_reward)
    error_text = f"Error: {error:.3f}"
    cv2.putText(img, error_text, (10, 120), font, font_scale, (255, 255, 255), thickness)

    return img


def add_text_label_on_top(image: np.ndarray, text: str) -> np.ndarray:
    """
    画像の上部にテキストラベルを追加する。
    Args:
        image (np.ndarray): 元の画像（RGB形式、uint8を想定）
        text (str): 追加するテキスト（変数名など）
    Returns:
        np.ndarray: テキストラベルが上部に追加された画像
    """
    # テキスト描画用のパラメータ
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    text_color = (255, 255, 255)
    bg_color = (50, 50, 50)

    # テキストのサイズを取得
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    # テキスト領域の高さ（余白を含む）
    # FFMPEGのlibx264エンコーダーは高さが2で割り切れる必要があるため、偶数に調整
    label_height = text_height + baseline + 10
    if label_height % 2 != 0:
        label_height += 1

    # テキスト用の画像領域を作成（画像と同じ幅）
    label_img = np.full((label_height, image.shape[1], 3), bg_color, dtype=np.uint8)

    # テキストを中央に配置
    text_x = 5
    text_y = text_height + 5
    cv2.putText(label_img, text, (text_x, text_y), font, font_scale, text_color, thickness)

    # テキスト領域と元の画像を縦に連結
    result = np.vstack((label_img, image))

    return result


def create_full_image_with_reward(
    env_render: np.ndarray,
    observation: np.ndarray,
    pred_image: np.ndarray,
    pred_reward: float,
    actual_reward: float,
) -> np.ndarray:
    """
    環境画像、観測画像、再構築画像、予測画像、報酬可視化を全て結合した画像を作成（RGB形式で返す）
    全ての画像をuint8に変換し、ラベルを追加する。
    """
    # 画像とラベル名を配列として定義
    images = [env_render, observation, pred_image]
    labels = ["env_render", "observation", "pred_image"]

    # uint8変換とラベル付与を一括処理
    labeled_images = [
        add_text_label_on_top(convert_to_uint8(img), label) for img, label in zip(images, labels)
    ]

    # 報酬可視化を作成してラベルを追加
    reward_vis = create_reward_visualization(pred_reward, actual_reward)
    reward_vis_labeled = add_text_label_on_top(reward_vis, "reward")

    # 全ての画像を連結
    final_image_bgr = concat_images(labeled_images + [reward_vis_labeled])
    final_image_rgb = cv2.cvtColor(final_image_bgr, cv2.COLOR_BGR2RGB)
    return final_image_rgb
