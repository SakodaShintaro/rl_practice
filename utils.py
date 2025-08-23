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
