import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import feature, measure, morphology
from sklearn.cluster import DBSCAN
import math

'''叶角判断参数设定'''
k1 = 20
k2 = 30
k3 = 40
k4 = 50

class ImprovedLeafDetector:
    def __init__(self):
        self.leaf_data = []
        self.debug_images = {}
    
    def preprocess_image(self, image_path):
        """图像预处理"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 多种颜色空间处理
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        
        # 创建绿色掩膜 (宽范围)
        mask1 = cv2.inRange(hsv, np.array([30, 20, 20]), np.array([90, 255, 255]))  # 宽绿色范围
        
        # 增加一个更明亮的绿色范围
        mask2 = cv2.inRange(hsv, np.array([30, 50, 100]), np.array([80, 255, 255]))
        
        # LAB空间中的绿色 (主要是a通道负值)
        a_channel = lab[:,:,1]
        _, mask3 = cv2.threshold(a_channel, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 组合掩膜
        mask = cv2.bitwise_or(mask1, mask2)
        mask = cv2.bitwise_or(mask, mask3)
        
        # 形态学操作改进
        kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # 先开运算去噪，再闭运算填充小孔
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel1, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel2, iterations=2)
        
        self.debug_images['mask'] = mask.copy()
        return img_rgb, mask
    
    def watershed_segmentation(self, mask, img_rgb):
        """修复的分水岭分割"""
        # 创建标记图像
        kernel = np.ones((3, 3), np.uint8)
        sure_bg = cv2.dilate(mask, kernel, iterations=3)
        
        # 距离变换
        dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        self.debug_images['dist_transform'] = dist_transform.copy()
        
        # 找到前景区域
        dist_max = dist_transform.max()
        _, sure_fg = cv2.threshold(dist_transform, 0.6*dist_max, 255, cv2.THRESH_BINARY)
        sure_fg = sure_fg.astype(np.uint8)
        
        # 使用局部极大值作为标记
        coords = feature.peak_local_max(
            dist_transform, 
            min_distance=20,  # 增加最小距离以避免过度分割
            threshold_abs=0.2*dist_max,
            exclude_border=False
        )
        
        # 创建标记图像
        markers = np.zeros_like(mask, dtype=np.int32)
        for i, (y, x) in enumerate(coords):
            # 使用1开始的标签 (0保留给背景)
            markers[y, x] = i + 1
        
        # 确保所有标记点在掩膜内部
        markers = markers * (mask > 0).astype(np.int32)
        
        # 若检测到的标记太少，则强制添加一些标记
        if np.max(markers) < 3 and np.sum(mask > 0) > 10000:
            # 基于形态学梯度找到可能的叶子区域
            gradient = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)
            _, thresh = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 找到梯度中的连通区域
            num_labels, labels = cv2.connectedComponents(thresh)
            
            # 为每个连通区域添加一个标记
            for label in range(1, num_labels):
                if np.sum(labels == label) > 100:  # 只考虑足够大的区域
                    y, x = np.where(labels == label)
                    if len(y) > 0:
                        idx = len(y) // 2  # 取中点
                        markers[y[idx], x[idx]] = np.max(markers) + 1
        
        # 为背景添加标记
        unknown = cv2.subtract(sure_bg, mask)
        markers[unknown == 255] = 0
        
        # 应用分水岭算法
        markers_copy = markers.copy() 
        cv2.watershed(img_rgb.astype(np.uint8), markers)
        
        self.debug_images['markers_before'] = markers_copy
        self.debug_images['markers_after'] = markers.copy()
        
        return markers
    
    def find_contours_from_markers(self, markers):
        """从分水岭标记中提取轮廓"""
        contours = []
        for label in np.unique(markers):
            if label <= 0:  # 跳过背景和边界
                continue
                
            # 创建该标记对应的二值图
            label_mask = np.zeros_like(markers, dtype=np.uint8)
            label_mask[markers == label] = 255
            
            # 进行形态学闭运算以平滑边界
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            label_mask = cv2.morphologyEx(label_mask, cv2.MORPH_CLOSE, kernel)
            
            # 提取轮廓
            region_contours, _ = cv2.findContours(label_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 添加有效轮廓
            for contour in region_contours:
                area = cv2.contourArea(contour)
                if area > 300:  # 面积阈值
                    contours.append(contour)
        
        return contours
    
    def contour_based_segmentation(self, mask):
        """基于轮廓的分割"""
        # 查找轮廓
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # 过滤和分析轮廓
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # 忽略太小的区域
            if area < 300:
                continue
                
            # 计算周长和轮廓复杂性
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
                
            # 形状特征
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            # 椭圆拟合
            if len(contour) >= 5:
                try:
                    (_, (width, height), _) = cv2.fitEllipse(contour)
                    aspect_ratio = max(width, height) / min(width, height) if min(width, height) > 0 else 0
                except:
                    aspect_ratio = 1
            else:
                aspect_ratio = 1
            
            # 计算凸包和凸度
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            convexity = area / hull_area if hull_area > 0 else 0
            
            # 叶子的特征过滤
            
            is_valid = (
                (circularity > 0.1 or aspect_ratio > 1.5) and  # 椭圆或不规则形状
                area > 300 and                              # 足够大
                convexity > 0.5                             # 凸度适中
            )
            
            if is_valid:
                valid_contours.append(contour)
        
        return valid_contours
    
    def split_large_contours(self, contours, original_mask):
        """分割大轮廓"""
        result = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # 如果面积太大，尝试分割
            if area > 5000:
                # 创建单独的掩膜
                mask = np.zeros_like(original_mask)
                cv2.drawContours(mask, [contour], -1, 255, -1)
                
                # 获取边界框
                x, y, w, h = cv2.boundingRect(contour)
                
                # 裁剪感兴趣区域
                roi_mask = mask[y:y+h, x:x+w]
                
                # 距离变换用于找分离点
                dist = cv2.distanceTransform(roi_mask, cv2.DIST_L2, 5)
                
                # 阈值处理找到分离区域
                _, sep_mask = cv2.threshold(dist, 0.5*dist.max(), 255, 0)
                sep_mask = sep_mask.astype(np.uint8)
                
                # 腐蚀操作
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                eroded = cv2.erode(sep_mask, kernel, iterations=1)
                
                # 查找分离后的组件
                num_labels, labels = cv2.connectedComponents(eroded)
                
                if num_labels > 2:  # 成功分离
                    for label in range(1, num_labels):
                        # 创建单个组件掩膜
                        comp_mask = np.zeros_like(labels, dtype=np.uint8)
                        comp_mask[labels == label] = 255
                        
                        # 膨胀回原始大小
                        comp_mask = cv2.dilate(comp_mask, kernel, iterations=2)
                        
                        # 将组件掩膜与原始轮廓掩膜相交
                        comp_mask = cv2.bitwise_and(comp_mask, roi_mask)
                        
                        # 提取新轮廓
                        sub_contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        
                        for sub_contour in sub_contours:
                            if cv2.contourArea(sub_contour) > 300:
                                # 调整轮廓坐标以匹配原始图像
                                adjusted_contour = sub_contour.copy()
                                adjusted_contour[:,:,0] += x
                                adjusted_contour[:,:,1] += y
                                result.append(adjusted_contour)
                else:
                    # 分割失败，保留原轮廓
                    result.append(contour)
            else:
                # 面积适中，不需要分割
                result.append(contour)
        
        return result
    
    def improved_leaf_detection(self, image_path):
        """改进的叶子检测算法"""
        # 1. 预处理
        img_rgb, mask = self.preprocess_image(image_path)
        
        # 2. 尝试分水岭分割
        markers = self.watershed_segmentation(mask, img_rgb)
        watershed_contours = self.find_contours_from_markers(markers)
        
        # 3. 基于轮廓的分割
        contour_based = self.contour_based_segmentation(mask)
        
        # 4. 分割大轮廓
        split_contours = self.split_large_contours(contour_based, mask)
        
        # 5. 合并结果
        all_contours = watershed_contours + split_contours
        
        # 6. 过滤和去重
        final_contours = self.filter_contours(all_contours, img_rgb.shape[:2])
        
        return final_contours, img_rgb, mask
    
    def filter_contours(self, contours, image_shape):
        """过滤和去重轮廓"""
        if not contours:
            return []
            
        # 过滤掉太小的轮廓
        filtered = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 200:
                continue
                
            # 计算边界框，确保不会太靠近边缘
            x, y, w, h = cv2.boundingRect(contour)
            border_margin = 5
            if (x <= border_margin or y <= border_margin or 
                x + w >= image_shape[1] - border_margin or 
                y + h >= image_shape[0] - border_margin):
                # 太靠近边缘，可能是不完整的叶子
                continue
                
            filtered.append(contour)
        
        # 去重：检测重叠的轮廓
        final = []
        mask = np.zeros(image_shape, dtype=np.uint8)
        
        # 按面积排序，优先保留较大的轮廓
        sorted_contours = sorted(filtered, key=cv2.contourArea, reverse=True)
        
        for contour in sorted_contours:
            # 检查是否与现有轮廓重叠过多
            temp_mask = np.zeros(image_shape, dtype=np.uint8)
            cv2.drawContours(temp_mask, [contour], -1, 1, -1)
            
            overlap = np.sum(temp_mask & mask)
            contour_area = np.sum(temp_mask)
            
            if overlap / contour_area < 0.5:  # 重叠不超过50%
                final.append(contour)
                # 更新掩膜
                mask |= temp_mask
        
        return final
    
    def find_leaf_axis_and_tip(self, contour, center):
        """找到叶子的长轴和叶尖"""
        if len(contour) < 5:
            return None, None, None
            
        # 椭圆拟合获取长轴方向
        try:
            ellipse = cv2.fitEllipse(contour)
            (ellipse_center, axes, angle) = ellipse
            
            # 计算长轴方向向量
            angle_rad = np.radians(angle)
            axis_vector = np.array([np.cos(angle_rad), np.sin(angle_rad)])
            
            # 计算长轴的两个端点
            major_axis = max(axes) / 2
            end1 = ellipse_center + major_axis * axis_vector
            end2 = ellipse_center - major_axis * axis_vector
            
            # 获取轮廓点
            contour_points = contour.reshape(-1, 2)
            
            # 计算每个轮廓点在长轴方向的投影
            # 将中心平移到原点
            centered_points = contour_points - np.array(center)
            
            # 投影到长轴方向
            projections = np.dot(centered_points, axis_vector)
            
            # 找到投影最大的点（叶尖）
            max_proj_idx = np.argmax(projections)
            min_proj_idx = np.argmin(projections)
            
            # 选择距离中心更远的点作为叶尖
            point_max = contour_points[max_proj_idx]
            point_min = contour_points[min_proj_idx]
            
            dist_max = np.linalg.norm(point_max - center)
            dist_min = np.linalg.norm(point_min - center)
            
            if dist_max > dist_min:
                tip = tuple(point_max.astype(int))
                tip_projection = projections[max_proj_idx]
            else:
                tip = tuple(point_min.astype(int))
                tip_projection = projections[min_proj_idx]
                
            return tip, axis_vector, tip_projection
            
        except:
            return None, None, None
    
    def calculate_leaf_droop_angle(self, center, tip, axis_vector):
        """计算叶尖相对于长轴的倾斜角度"""
        if tip is None or axis_vector is None:
            return None
            
        # 计算从中心到叶尖的向量
        tip_vector = np.array(tip) - np.array(center)
        
        # 标准化向量
        tip_vector_norm = tip_vector / np.linalg.norm(tip_vector)
        axis_vector_norm = axis_vector / np.linalg.norm(axis_vector)
        
        # 计算两个向量的夹角
        dot_product = np.dot(tip_vector_norm, axis_vector_norm)
        
        # 防止数值误差导致的域错误
        dot_product = np.clip(dot_product, -1.0, 1.0)
        
        # 计算角度（弧度）
        angle_rad = np.arccos(dot_product)
        
        # 转换为角度
        angle_deg = np.degrees(angle_rad)
        

        if angle_deg > 90:
            angle_deg = 180 - angle_deg
            
        return angle_deg
    
    def analyze_leaves(self, image_path, visualize=True):
        """分析叶子"""
        # 检测叶子
        leaf_contours, img_rgb, mask = self.improved_leaf_detection(image_path)
        
        if not leaf_contours:
            return "未检测到叶子", [], img_rgb
        
        result_img = img_rgb.copy()
        leaf_angles = []
        self.leaf_data = []
        
        # 为每个轮廓分配唯一颜色
        colors = []
        for _ in range(len(leaf_contours)):
            colors.append((
                np.random.randint(50, 255),
                np.random.randint(50, 255),
                np.random.randint(50, 255)
            ))
        
        # 分析每个叶子
        for i, contour in enumerate(leaf_contours):
            # 计算中心
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
                
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            center = (cx, cy)
            
            # 找到叶子的长轴和叶尖
            tip, axis_vector, tip_projection = self.find_leaf_axis_and_tip(contour, center)
            
            if tip is None or axis_vector is None:
                continue
                
            # 计算叶尖相对于长轴的角度
            angle = self.calculate_leaf_droop_angle(center, tip, axis_vector)
            
            if angle is None:
                continue
            
            leaf_angles.append(angle)
            
            # 存储叶子数据
            leaf_info = {
                'leaf_id': i + 1,
                'center': center,
                'tip': tip,
                'angle': angle,
                'area': cv2.contourArea(contour),
                'axis_vector': axis_vector
            }
            self.leaf_data.append(leaf_info)
            
            if visualize:
                # 使用唯一颜色绘制轮廓
                color = colors[i]
                cv2.drawContours(result_img, [contour], -1, color, 2)
                
                # 绘制中心
                cv2.circle(result_img, center, 3, (0, 0, 255), -1)
                
                # 绘制叶尖
                cv2.circle(result_img, tip, 3, (255, 0, 0), -1)
                
                # 绘制长轴（参考线）
                axis_length = 50
                axis_end = (
                    int(center[0] + axis_length * axis_vector[0]),
                    int(center[1] + axis_length * axis_vector[1])
                )
                cv2.line(result_img, center, axis_end, (0, 255, 255), 2)  # 黄色长轴
                
                # 绘制从中心到叶尖的线（实际测量线）
                cv2.line(result_img, center, tip, (255, 255, 0), 2)  # 青色测量线
                
                # 显示叶子编号
                label_pos = (center[0] - 10, center[1] - 10)
                cv2.putText(result_img, f'{i+1}', 
                          label_pos, cv2.FONT_HERSHEY_SIMPLEX, 
                          0.5, (255, 255, 255), 2)
                
                # 显示角度
                angle_pos = (tip[0] + 5, tip[1] - 5)
                cv2.putText(result_img, f'{angle:.1f}°', 
                          angle_pos, cv2.FONT_HERSHEY_SIMPLEX, 
                          0.4, (255, 255, 255), 1)
        
        # 分析健康状态
        if leaf_angles:
            avg_angle = np.mean(leaf_angles)
            status = self.determine_health_status(avg_angle, leaf_angles)
        else:
            status = "无法分析"
            
        return status, leaf_angles, result_img
    
    def determine_health_status(self, avg_angle, angles):
        """确定健康状态 - 基于与长轴的偏离角度"""
        
        severe_count = sum(1 for a in angles if a > 50)  # 相对于长轴偏离超过50度认为严重
        severe_ratio = severe_count / len(angles)
        
        if avg_angle < k1 and severe_ratio < 0.2:
            return "植物状态优秀 - 叶子沿长轴挺立"
        elif avg_angle < k2 and severe_ratio < 0.3:
            return "植物状态良好 - 叶子基本挺立"
        elif avg_angle < k3 and severe_ratio < 0.5:
            return "轻微缺水 - 叶子开始偏离长轴"
        elif avg_angle < k4 and severe_ratio < 0.7:
            return "中度缺水 - 叶子明显偏离长轴"
        else:
            return "严重缺水 - 叶子大幅偏离长轴方向"
    
    def generate_report(self):
        """生成详细分析报告"""
        if not self.leaf_data:
            return "没有分析数据"
            
        angles = [leaf['angle'] for leaf in self.leaf_data]
        
        report = "详细叶子分析:\n"
        report += "=" * 40 + "\n"
        
        for leaf in self.leaf_data:
            report += f"叶子 {leaf['leaf_id']}: 相对长轴偏离角度 {leaf['angle']:.1f}°\n"
        
        report += "\n=== 统计数据 ===\n"
        report += f"检测到叶子数量: {len(angles)}\n"
        report += f"平均偏离角度: {np.mean(angles):.1f}°\n"
        report += f"最大偏离角度: {max(angles):.1f}°\n"
        report += f"最小偏离角度: {min(angles):.1f}°\n"
        report += f"标准差: {np.std(angles):.1f}°\n"
        
        # 角度分布（相对于长轴）
        ranges = [0, 10, 15, 20, 30, 45, 90]
        report += "\n=== 长轴偏离角度分布 ===\n"
        
        for i in range(len(ranges)-1):
            lower = ranges[i]
            upper = ranges[i+1]
            count = sum(1 for a in angles if lower <= a < upper)
            percentage = count / len(angles) * 100
            
            if lower < k1:
                status = "优秀"
            elif lower < k2:
                status = "良好"
            elif lower < k3:
                status = "轻微缺水"
            elif lower < k4:
                status = "中度缺水"
            else:
                status = "严重缺水"
                
            report += f"{lower}-{upper}°: {count} 片叶子 ({percentage:.1f}%) - {status}\n"
        
        return report
    
    def show_debug_info(self):
        """显示调试信息"""
        if not self.debug_images:
            return
            
        plt.figure(figsize=(15, 10))
        
        # 显示可用的调试图像
        image_names = list(self.debug_images.keys())
        n_images = len(image_names)
        
        # 计算行列数
        cols = min(3, n_images)
        rows = (n_images + cols - 1) // cols
        
        for i, name in enumerate(image_names):
            plt.subplot(rows, cols, i+1)
            
            img = self.debug_images[name]
            
            if name == 'dist_transform':
                plt.imshow(img, cmap='hot')
            elif 'markers' in name:
                plt.imshow(img, cmap='nipy_spectral')
            else:
                if len(img.shape) == 2:
                    plt.imshow(img, cmap='gray')
                else:
                    plt.imshow(img)
                    
            plt.title(name)
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()