import cv2
import numpy as np


def concat_images(main: np.ndarray, sub_image_list: list[np.ndarray]) -> np.ndarray:
    """
    main画像の右にsub_image_listを並べて配置し、余白をpadで埋める。

    [main00][main01]...[main0w][sub_image_list]
    ...
    [mainh0][mainh2]...[mainhw][pad.]

    Args:
        main (np.ndarray,  (400, 600, 3)): The main image.
        sub_image_list (list[np.ndarray]): The list of sub-images to concatenate.
    Returns:
        np.ndarray: The concatenated image.
    """
    concat = cv2.vconcat(sub_image_list)
    rem = main.shape[0] - concat.shape[0]
    concat = cv2.copyMakeBorder(concat, 0, rem, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    concat *= 255
    concat = np.clip(concat, 0, 255).astype(np.uint8)
    concat = cv2.hconcat([main, concat])
    bgr_array = cv2.cvtColor(concat, cv2.COLOR_RGB2BGR)
    return bgr_array


def create_reward_visualization(pred_reward: float, actual_reward: float) -> np.ndarray:
    """
    報酬予測と実際の報酬を可視化する関数（シンプルなテキスト表示）
    """
    height, width = (500, 200)
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
    # まず既存の画像を連結
    first_concat = concat_images(env_render, [obs_for_render, reconstruction, pred_image])

    # 報酬可視化を作成してメイン画像の高さに合わせる
    reward_vis = create_reward_visualization(pred_reward, actual_reward)
    reward_vis_resized = cv2.resize(reward_vis, (reward_vis.shape[1], first_concat.shape[0]))
    reward_vis_normalized = reward_vis_resized.astype(np.float32) / 255.0

    # 2回目のconcat: first_concatの右側に報酬可視化を追加
    # ここではRGB形式のままで返す
    concat = cv2.vconcat([reward_vis_normalized])
    rem = first_concat.shape[0] - concat.shape[0]
    if rem >= 0:
        concat = cv2.copyMakeBorder(concat, 0, rem, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    concat *= 255
    concat = np.clip(concat, 0, 255).astype(np.uint8)

    # first_concatをBGRからRGBに変換してから結合
    first_concat_rgb = cv2.cvtColor(first_concat, cv2.COLOR_BGR2RGB)
    final_image_rgb = cv2.hconcat([first_concat_rgb, concat])

    return final_image_rgb
