# Alpha-Channel Based Physical Haze Synthesis (AC-PHS)
import os
import cv2
import pyzjr
import random
import numpy as np

def random_addfog(
        background,
        overlay_path=r"E:\PythonProject\img_processing_techniques_main\filter_image\cloud_512_thin",
        a=0.75, b=1.2,
        sigma_s=0.15,
        sigma_v=0.85,
        layers=1,
        prob=0.5,
):
    image_path_list = pyzjr.get_image_path(overlay_path)
    fused_image = background.copy()
    # Group cloud images by category (assuming file name format is 'category_number.png')
    fog_groups = {}
    for path in image_path_list:
        base_name = os.path.basename(path)
        fog_class = base_name.split('_')[0]  # Extract category prefix (e.g., fog1)
        if fog_class not in fog_groups:
            fog_groups[fog_class] = []
        fog_groups[fog_class].append(path)

    # Randomly select cloud images from different categories
    selected_classes = random.sample(list(fog_groups.keys()), min(layers, len(fog_groups)))

    for fog_class in selected_classes:
        rand_path = random.choice(fog_groups[fog_class])
        rand_float = random.uniform(a, b)
        overlay = cv2.imread(rand_path, cv2.IMREAD_UNCHANGED)
        if overlay.shape[2] == 4:  # Ensure it is an RGBA image
            # Converting to HSV color space significantly reduces saturation and decreases brightness.
            hsv = cv2.cvtColor(overlay[..., :3], cv2.COLOR_BGR2HSV)
            hsv[..., 1] = hsv[..., 1] * sigma_s  # Reduce saturation to sigma_s
            hsv[..., 2] = hsv[..., 2] * sigma_v  # Reduce the brightness to sigma_v
            # Increase the proportion of gray components
            gray = cv2.cvtColor(overlay[..., :3], cv2.COLOR_BGR2GRAY)
            overlay[..., :3] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR) * 0.70 + gray[..., np.newaxis] * 0.30
        overlay = np.flip(overlay, 0)
        if random.random() > prob:
            overlay = np.flip(overlay, 1)
        fused_image = pyzjr.OverlayPng(fused_image, overlay, alpha_gain=rand_float)
        print(f"Use transparency gain {rand_float:.2f} combined {os.path.basename(rand_path)}")
    return fused_image


def random_addfog_folder(
        background_path,
        save_path=r'./results',
        overlay_path=r"E:\PythonProject\img_processing_techniques_main\filter_image\cloud_512_thin",
        a=0.86,
        b=1.3,
        sigma_s=0.15,
        sigma_v=0.85,
        layers=3
):
    os.makedirs(save_path, exist_ok=True)
    bg_image_path_list = pyzjr.get_image_path(background_path)
    for i in range(len(bg_image_path_list)):
        background = cv2.imread(bg_image_path_list[i])
        fused_image = random_addfog(background,
                                    overlay_path,
                                    a=a, b=b,
                                    sigma_s=sigma_s,
                                    sigma_v=sigma_v,
                                    layers=layers)
        new_path = os.path.join(save_path, os.path.basename(bg_image_path_list[i]))
        print(f"{i + 1}: {bg_image_path_list[i]} -> {new_path}")
        cv2.imwrite(new_path, fused_image)


if __name__=="__main__":
    # background_path = r'E:\PythonProject\img_processing_techniques_main\filter_image\LEVIR_DEHAZE\thick\GT'
    save_path = r'./results'
    overlay_cloudmist_path = r'E:\PythonProject\dehazeprojects/data\cloud262\cloud_shape_512'
    # random_addfog_folder(background_path, save_path, overlay_path=overlay_cloudmist_path)
    background_path = r'E:\PythonProject\dehazeprojects/assert\ship_344.png'
    background = cv2.imread(background_path)
    fused_image = random_addfog(background, layers=2)
    cv2.imshow('Fused Image', fused_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


