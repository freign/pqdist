import matplotlib.pyplot as plt
import numpy as np




exact_times = [205, 355, 716, 1305, 2589, 5143]
naive_times = [37, 56, 98, 158, 300, 575]
simd_times = [19, 22, 30, 41, 67, 118]
x = np.arange(1, 7) 
x_labels = [2** i for i in range(9, 15)]

plt.figure(figsize=(12, 7), dpi=100)


exact_line, = plt.plot(x, exact_times, 
                      color='#2E86C1', 
                      marker='o', 
                      markersize=10,
                      markerfacecolor='white',
                      markeredgewidth=2,
                      linestyle='--',
                      linewidth=2.5,
                      label='Exact Distance Computation')

naive_line, = plt.plot(x, naive_times,
                      color='#E67E22',
                      marker='s',
                      markersize=10,
                      markerfacecolor='white',
                      markeredgewidth=2,
                      linestyle='-.',
                      linewidth=2.5,
                      label='Naive PQ Distance Computation')

simd_line, = plt.plot(x, simd_times,
                     color='#27AE60',
                     marker='^',
                     markersize=10,
                     markerfacecolor='white',
                     markeredgewidth=2,
                     linestyle=':',
                     linewidth=2.5,
                     label='SIMD-accelerated PQ Distance Computation')

def add_labels(data_points, color, y_offset_ratio=0.15):
    """自动添加带偏移量的数据标签"""
    for i, (xi, yi) in enumerate(zip(x, data_points)):
        offset = yi * y_offset_ratio  # 动态偏移量（按数值比例）
        plt.text(xi, yi + offset, f'{yi:,}',  # 使用千位分隔符
                ha='center', va='bottom',
                color=color, fontsize=9,
                bbox=dict(boxstyle='round,pad=0.2', 
                         facecolor='white', 
                         edgecolor=color, 
                         alpha=0.8))

# 应用标签（调整不同曲线的偏移比例）
add_labels(exact_times, '#2E86C1', y_offset_ratio=0.08)  # 精确距离数值较大，减小偏移
add_labels(naive_times, '#E67E22', y_offset_ratio=0.15)
add_labels(simd_times, '#27AE60', y_offset_ratio=0.15)   # SIMD数值小，增大偏移

plt.title('Computation Time Comparison on SIFT Dataset\n(PQ Configuration: 16-subspaces, 4-nbits)', 
         fontsize=14, pad=20)
plt.xlabel('Data Scale', fontsize=12)
plt.ylabel('Time (ms)', fontsize=12)
plt.xticks(x, x_labels)
plt.yscale('log')  


plt.grid(True, which='both', linestyle='--', alpha=0.7)



plt.legend(handles=[exact_line, naive_line, simd_line],
          loc='upper left',
          frameon=True,
          shadow=True,
          fancybox=True)


plt.tight_layout()
plt.show()


plt.savefig('sift.png', dpi=300, bbox_inches='tight')