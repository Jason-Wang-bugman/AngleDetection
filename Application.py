import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import feature, measure, morphology
from sklearn.cluster import DBSCAN
import math

import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
matplotlib.rcParams['axes.unicode_minus'] = False   # 解决负号显示问题

def main():
    detector = ImprovedLeafDetector()
    
    image_path = "D:\\standard_tree_test4.png"  
    try:
        status, angles, result_img = detector.analyze_leaves(image_path, visualize=True)
        
        print(f"\n=== 分析结果 ===")
        print(f"植物状态: {status}")
        print(f"检测到叶子数量: {len(angles)}")
        
        if angles:
            print(f"平均长轴偏离角度: {np.mean(angles):.1f}°")
            print(f"角度范围: {min(angles):.1f}° - {max(angles):.1f}°")
            print(f"标准差: {np.std(angles):.1f}°")
        
        # 显示详细报告
        print("\n" + detector.generate_report())
        
        # 显示结果
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        original = cv2.imread(image_path)
        plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        plt.title("原始图像")
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(result_img)
        plt.title(f"叶子检测结果 ({len(angles)}片)\n{status}")
        plt.axis('off')
        
        if angles:
            plt.subplot(1, 3, 3)
            plt.hist(angles, bins=10, range=(0, 90), alpha=0.7, color='green')
            plt.axvline(np.mean(angles), color='red', linestyle='--', 
                       label=f'平均: {np.mean(angles):.1f}°')
            plt.xlabel('相对长轴偏离角度 (度)')
            plt.ylabel('叶子数量')
            plt.title('长轴偏离角度分布')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # 显示调试信息
        detector.show_debug_info()
        
    except Exception as e:
        print(f"处理出错: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()