import cv2
import numpy as np


def concat_images(main, sub1, sub2):
    """
    main画像の右上にsub1とsub2を並べて配置し、余白をpadで埋める。

    [main00][main01]...[main0w][sub1]
    [main10][main11]...[main1w][sub2]
    ...
    [mainh0][mainh2]...[mainhw][pad.]

    Args:
        main (np.ndarray,  (400, 600, 3)): The main image.
        sub1 (np.ndarray), (96, 96, 3)): The first sub-image to concatenate.
        sub2 (np.ndarray), (96, 96, 3)): The second sub-image to concatenate.
    Returns:
        np.ndarray: The concatenated image.
    """
    bgr_array = cv2.cvtColor(main, cv2.COLOR_RGB2BGR)
    return bgr_array
    concat = cv2.vconcat([sub1, sub2])  # (192, 96, 3)
    rem = main.shape[0] - concat.shape[0]
    concat = cv2.copyMakeBorder(concat, 0, rem, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    concat *= 255
    concat = np.clip(concat, 0, 255).astype(np.uint8)
    concat = cv2.hconcat([main, concat])  # (400, 696, 3)
    bgr_array = cv2.cvtColor(concat, cv2.COLOR_RGB2BGR)
    return bgr_array  # (400, 696, 3)
