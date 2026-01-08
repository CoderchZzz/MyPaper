import numpy as np
import matplotlib.pyplot as plt

# 如果你不需要弹窗，可以设置非交互式后端（可选，防止报错）
import matplotlib

matplotlib.use('Agg')


def plot_simulated_affinity_matrix(size=50, sigma=8, cmap='Blues'):
    """
    绘制并保存模拟的亲和力矩阵图片
    """
    # 1. 生成网格坐标
    x = np.arange(size)
    y = np.arange(size)
    X, Y = np.meshgrid(x, y)

    # 2. 计算距离
    distance = np.abs(X - Y)

    # 3. 转换为亲和力数值
    affinity_matrix = np.exp(-distance / sigma)

    # 4. 绘图
    plt.figure(figsize=(8, 8))
    plt.imshow(affinity_matrix, cmap=cmap, origin='upper', interpolation='nearest')

    # 颜色条
    cbar = plt.colorbar(shrink=0.8)
    cbar.set_label('Affinity / Similarity', rotation=270, labelpad=15)

    plt.title('Simulated Affinity Matrix', fontsize=16, pad=20)
    plt.axis('off')

    # --- 修改点：使用 savefig 替代 show ---
    save_path = 'affinity_matrix.png'
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"图片已保存为: {save_path}")
    # ------------------------------------


# 运行
plot_simulated_affinity_matrix(size=20, sigma=6, cmap='Blues')