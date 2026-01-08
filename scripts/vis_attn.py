import torch
import clip
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# 导入你项目中的工具函数
from clip.clip_tool import generate_clip_fts

# 设置非交互式后端
import matplotlib

matplotlib.use('Agg')


def save_individual_attention_maps(img_path, model_path, layers_to_viz=[1, 4, 7, 11], device='cuda'):
    """
    分别保存指定层的 CLIP 注意力图为单独的图片文件。
    """
    # 1. 加载模型
    print(f"Loading CLIP model from {model_path}...")
    # 注意：如果你的路径包含中文，OpenCV读取可能会有问题，但这里用的是 PIL 和 torch.load，通常没问题
    try:
        model, preprocess = clip.load(model_path, device=device)
    except Exception as e:
        print(f"加载模型失败，请检查路径: {e}")
        return

    model.eval()

    # 2. 加载和预处理图片
    if not os.path.exists(img_path):
        print(f"Error: Image not found at {img_path}")
        return

    # 尝试加载图片
    try:
        ori_image = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"无法打开图片: {e}")
        return

    w, h = ori_image.size
    image_tensor = preprocess(ori_image).unsqueeze(0).to(device)

    # 3. 提取所有层的注意力权重
    print("Extracting attention weights...")
    with torch.no_grad():
        # generate_clip_fts 返回: image_features_all, attn_weight_list
        # attn_weight_list 是一个列表，包含每一层的 [Batch, Tokens, Tokens]
        _, attn_weight_list = generate_clip_fts(image_tensor, model, require_all_fts=True)

    # 获取基础文件名，用于保存结果
    # 处理路径中的中文和特殊字符，最好只取文件名
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    output_dir = "attn_maps_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Saving images to folder: {output_dir}")

    # 4. 遍历指定层并保存
    for layer_idx in layers_to_viz:
        # 检查层索引是否越界
        if layer_idx >= len(attn_weight_list):
            print(f"Warning: Layer {layer_idx} out of bounds (max {len(attn_weight_list) - 1}), skipping.")
            continue

        # 获取指定层的注意力矩阵 [Batch, Tokens, Tokens] -> [Tokens, Tokens]
        # 此时 attn_weight_list 里的 tensor 已经是 float16 或 float32
        attn_matrix = attn_weight_list[layer_idx][0]

        # 提取 [CLS] token (索引0) 对所有 patch tokens (索引1:) 的注意力
        # Shape: [Num_Patches]
        cls_attn = attn_matrix[0, 1:]

        # 恢复成 2D 形状
        # grid_size = sqrt(N_patches)
        grid_size = int(np.sqrt(cls_attn.shape[0]))
        cls_attn_2d = cls_attn.reshape(grid_size, grid_size)

        # 转为 numpy
        cls_attn_np = cls_attn_2d.cpu().float().numpy()

        # 缩放到原图大小 (Resize)
        # 注意 cv2.resize 接收 (width, height)
        attn_map_resized = cv2.resize(cls_attn_np, (w, h))

        # 归一化到 0-1 (这对热力图可视化非常重要)
        min_val, max_val = attn_map_resized.min(), attn_map_resized.max()
        if max_val - min_val > 0:
            attn_map_norm = (attn_map_resized - min_val) / (max_val - min_val)
        else:
            attn_map_norm = attn_map_resized  # 避免除零

        # --- 核心修改：单独保存 ---
        save_filename = os.path.join(output_dir, f"{base_name}_layer{layer_idx}.png")

        # 使用 plt.imsave 直接保存数组为图片，自动应用 colormap
        # cmap='jet' 是经典的彩虹热力图，也可以换成 'viridis' 或 'inferno'
        plt.imsave(save_filename, attn_map_norm, cmap='jet')
        print(f"Saved: {save_filename}")

    print("Done! All attention maps saved.")


if __name__ == "__main__":
    # 配置你的路径
    # 注意：Windows路径最好在字符串前加 r，或者把 \ 改成 /
    IMAGE_PATH = r"E:\研究生论文\写论文\论文插图/2007_001825.jpg"
    MODEL_PATH = r"E:\浏览器下载\code\CLIP-ES-mainPro\pretrained_models\clip\ViT-B-16.pt"

    # 选择要可视化的层
    # LAYERS = [0,1,2,3,4,5,6,7,8,9,10]
    LAYERS = [1, 4, 7, 10]

    save_individual_attention_maps(IMAGE_PATH, MODEL_PATH, layers_to_viz=LAYERS)