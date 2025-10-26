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
        if img.dtype == np.float32:
            img *= 255.0
            img = np.clip(img, 0, 255).astype(np.uint8)
        if img.shape[0] < max_height:
            padding = np.zeros((max_height - img.shape[0], img.shape[1], 3), dtype=img.dtype)
            padded_img = np.vstack((img, padding))
        else:
            padded_img = img
        padded_images.append(padded_img)
    concat = cv2.hconcat(padded_images)
    bgr_array = cv2.cvtColor(concat, cv2.COLOR_RGB2BGR)
    return bgr_array


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


def create_full_image_with_reward(
    env_render: np.ndarray,
    obs_for_render: np.ndarray,
    reconstruction: np.ndarray,
    pred_image: np.ndarray,
    pred_reward: float,
    actual_reward: float,
) -> np.ndarray:
    """
    環境画像、観測画像、再構築画像、予測画像、報酬可視化を全て結合した画像を作成（RGB形式で返す）
    """
    reward_vis = create_reward_visualization(pred_reward, actual_reward)
    final_image_bgr = concat_images(
        [env_render, obs_for_render, reconstruction, pred_image, reward_vis]
    )
    final_image_rgb = cv2.cvtColor(final_image_bgr, cv2.COLOR_BGR2RGB)
    return final_image_rgb
