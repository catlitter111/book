# -*- coding: utf-8 -*-
"""
简化的双目立体视觉系统 (增强版)
===================
基于OpenCV和Open3D的双目立体视觉程序
- 校正图像处理
- SGBM立体匹配+WLS滤波
- 3D点云生成
- 距离测量功能
- 深度图显示 (新增功能)

作者: C211学生
little cat
little sss
优化: 添加WLS滤波以改善视差图质量，添加深度图显示
cat little
"""

import cv2
import numpy as np
import os
import time
import sys
import threading
import math
from PIL import Image, ImageDraw, ImageFont
import open3d as o3d
from functools import lru_cache


# -------------------------------- 配置参数类 --------------------------------
class StereoConfig:
    """立体视觉系统配置类"""

    def __init__(self):
        # 相机内参和外参
        # self.baseline = 60.98  # 基线距离
        # self.focal_length = 446  # 焦距
        # self.cx = 295  # 光心x坐标
        # self.cy = 246  # 光心y坐标

        self.baseline = 25.100  # 基线距离
        self.focal_length = 663  # 焦距
        self.cx = 317  # 光心x坐标
        self.cy = 210  # 光心y坐标

        # SGBM算法参数
        self.minDisparity = 3
        self.numDisparities = 16  # 必须是16的倍数
        self.blockSize = 7
        self.P1 = 1176
        self.P2 = 4704
        self.disp12MaxDiff = 4
        self.preFilterCap = 31
        self.uniquenessRatio = 10
        self.speckleWindowSize = 100
        self.speckleRange = 32

        # # SGBM算法参数
        # self.minDisparity = 5
        # self.numDisparities = 112  # 必须是16的倍数
        # self.blockSize = 15
        # self.P1 = 3375
        # self.P2 = 5400
        # self.disp12MaxDiff = 8
        # self.preFilterCap = 31
        # self.uniquenessRatio = 10
        # self.speckleWindowSize = 100
        # self.speckleRange = 32
        self.mode = cv2.STEREO_SGBM_MODE_SGBM_3WAY

        # WLS滤波器参数
        self.wls_lambda = 8000.0  # 滤波强度
        self.wls_sigma = 1.5  # 颜色相似性敏感度

        # 深度筛选范围
        self.min_disparity_threshold = 1100
        self.max_disparity_threshold = 1570

        # 点云参数
        self.min_distance_mm = 100.0
        self.max_distance_mm = 10000.0
        self.max_x_mm = 5000.0
        self.max_y_mm = 5000.0
        self.scale_correction = 1.0

        # 显示控制
        self.window_names = {
            "original": "original",
            "rectified": "rectified",
            "disparity": "disparity",
            "depth": "depth"  # 新增深度图窗口
        }

        # 深度图参数
        self.depth_min_display = 100.0  # 最小显示深度(毫米)
        self.depth_max_display = 5000.0  # 最大显示深度(毫米)
        self.depth_colormap = cv2.COLORMAP_INFERNO  # 深度图颜色映射方案

        # 相机参数
        self.camera_id = 1
        self.frame_width = 1280
        self.frame_height = 480
        # self.frame_width = 1920
        # self.frame_height = 1080
        self.fps_limit = 30

        # 测距功能
        self.enable_click_measure = True
        self.measure_points = []
        self.max_measure_points = 10

        # 临时存储视差图尺寸，供鼠标回调使用
        self.last_disp_size = (0, 0)

        # 输出目录
        self.output_dir = "output"
        os.makedirs(self.output_dir, exist_ok=True)


# -------------------------------- 相机校正配置 --------------------------------
class stereoCamera(object):
    """双目相机参数类"""

    def __init__(self):
        # 左相机内参
        self.cam_matrix_left = np.array([[660.1946, 0, 326.3185], [0, 660.8720, 207.1556], [0, 0, 1]])

        # 右相机内参
        self.cam_matrix_right = np.array([[665.1635, 0, 319.9729], [0, 665.7919, 212.9630], [0, 0, 1]])

        # 左右相机畸变系数:[k1, k2, p1, p2, k3]
        self.distortion_l = np.array([[-0.0682, 0.1546, 0, 0, 0]])
        self.distortion_r = np.array([[-0.0749, 0.1684, 0, 0, 0]])

        # 旋转矩阵
        self.R = np.array([[1.0, 6.140854786327222e-04, -0.0022],
                           [-6.240288417695294e-04, 1, -0.0046],
                           [0.0022, 0.0046, 1]])

        # 平移矩阵 - 转换为列向量形式
        self.T = np.array([[-25.0961], [-0.0869], [-0.1893]])

        # 焦距和基线距离
        self.focal_length = 663
        self.baseline = abs(self.T[0][0])

        # Q矩阵（视差到深度的映射矩阵）
        self.Q = None  # 在getRectifyTransform中计算


# -------------------------------- 图像处理函数 --------------------------------
def preprocess(img1, img2):
    """
    图像预处理：将彩色图转为灰度图并应用直方图均衡化以增强对比度
    """
    # 转换为灰度图
    if img1.ndim == 3:
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    else:
        img1_gray = img1.copy()

    if img2.ndim == 3:
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        img2_gray = img2.copy()

    # 应用CLAHE增强对比度
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img1_eq = clahe.apply(img1_gray)
    img2_eq = clahe.apply(img2_gray)

    return img1_eq, img2_eq


def undistortion(image, camera_matrix, dist_coeff):
    """
    消除图像畸变，使用相机内参和畸变系数
    """
    h, w = image.shape[:2]
    # 计算最佳新相机矩阵
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeff, (w, h), 1, (w, h))
    # 校正畸变
    undistorted = cv2.undistort(image, camera_matrix, dist_coeff, None, new_camera_matrix)

    # 如果ROI有效，裁剪图像
    x, y, w, h = roi
    if w > 0 and h > 0:
        undistorted = undistorted[y:y + h, x:x + w]

    return undistorted


def getRectifyTransform(height, width, config):
    """
    获取畸变校正和立体校正的映射变换矩阵及重投影矩阵
    """
    left_K = config.cam_matrix_left
    right_K = config.cam_matrix_right
    left_dist = config.distortion_l
    right_dist = config.distortion_r
    R = config.R
    T = config.T

    # 计算立体校正参数
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        left_K, left_dist, right_K, right_dist,
        (width, height), R, T,
        flags=cv2.CALIB_ZERO_DISPARITY,  # 使校正图像的主点具有相同的像素坐标
        alpha=0.5  # 0表示裁剪所有不相交的像素，1表示保留所有像素
    )

    # 保存Q矩阵
    config.Q = Q

    # 生成映射矩阵
    map1x, map1y = cv2.initUndistortRectifyMap(
        left_K, left_dist, R1, P1, (width, height), cv2.CV_32FC1
    )
    map2x, map2y = cv2.initUndistortRectifyMap(
        right_K, right_dist, R2, P2, (width, height), cv2.CV_32FC1
    )

    return map1x, map1y, map2x, map2y, Q


def rectifyImage(image1, image2, map1x, map1y, map2x, map2y):
    """
    对图像应用畸变校正和立体校正
    """
    rectified_img1 = cv2.remap(image1, map1x, map1y, cv2.INTER_AREA)
    rectified_img2 = cv2.remap(image2, map2x, map2y, cv2.INTER_AREA)
    return rectified_img1, rectified_img2


def draw_line(image1, image2):
    """
    在校正后的图像上绘制平行线，验证立体校正效果
    """
    height = max(image1.shape[0], image2.shape[0])
    width = image1.shape[1] + image2.shape[1]
    output = np.zeros((height, width, 3), dtype=np.uint8)  # 创建空白画布

    # 确保图像是彩色的
    img1_color = image1 if image1.ndim == 3 else cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
    img2_color = image2 if image2.ndim == 3 else cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)

    output[0:image1.shape[0], 0:image1.shape[1]] = img1_color
    output[0:image2.shape[0], image1.shape[1]:] = img2_color

    # 绘制水平线，间隔30像素
    for k in range(height // 30):
        y = 30 * (k + 1)
        cv2.line(output, (0, y), (2 * width, y), (0, 255, 0), thickness=1, lineType=cv2.LINE_AA)

    # 在图像中间位置添加垂直分隔线
    mid_x = image1.shape[1]
    cv2.line(output, (mid_x, 0), (mid_x, height), (0, 0, 255), thickness=2, lineType=cv2.LINE_AA)

    return output


def stereoMatchSGBM_WLS(left_image, right_image, config):
    """
    使用SGBM算法结合WLS滤波器计算高质量视差图
    """
    # 导入ximgproc模块
    from cv2 import ximgproc

    # 确保输入图像是灰度图
    if left_image.ndim != 2:
        left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
    if right_image.ndim != 2:
        right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

    # 创建左图的SGBM匹配器
    left_matcher = cv2.StereoSGBM_create(
        minDisparity=config.minDisparity,
        numDisparities=config.numDisparities,
        blockSize=config.blockSize,
        P1=config.P1,
        P2=config.P2,
        disp12MaxDiff=config.disp12MaxDiff,
        preFilterCap=config.preFilterCap,
        uniquenessRatio=config.uniquenessRatio,
        speckleWindowSize=config.speckleWindowSize,
        speckleRange=config.speckleRange,
        mode=config.mode
    )

    # 创建右图的匹配器（将minDisparity参数取反）
    right_matcher = cv2.StereoSGBM_create(
        minDisparity=-config.numDisparities + config.minDisparity,
        numDisparities=config.numDisparities,
        blockSize=config.blockSize,
        P1=config.P1,
        P2=config.P2,
        disp12MaxDiff=config.disp12MaxDiff,
        preFilterCap=config.preFilterCap,
        uniquenessRatio=config.uniquenessRatio,
        speckleWindowSize=config.speckleWindowSize,
        speckleRange=config.speckleRange,
        mode=config.mode
    )

    # 计算左右视差图
    left_disp = left_matcher.compute(left_image, right_image)
    right_disp = right_matcher.compute(right_image, left_image)

    # 转换为浮点数
    left_disp = left_disp.astype(np.float32) / 16.0
    right_disp = right_disp.astype(np.float32) / 16.0

    # 创建WLS滤波器
    wls_filter = ximgproc.createDisparityWLSFilter(left_matcher)

    # 设置WLS滤波器参数
    wls_filter.setLambda(config.wls_lambda)  # 平滑程度
    wls_filter.setSigmaColor(config.wls_sigma)  # 颜色相似性敏感度

    # 应用WLS滤波器
    filtered_disp = wls_filter.filter(left_disp, left_image, disparity_map_right=right_disp)

    # 应用形态学处理改善质量
    kernel = np.ones((3, 3), np.uint8)
    filtered_disp = cv2.morphologyEx(filtered_disp, cv2.MORPH_CLOSE, kernel)

    # 过滤小视差值和无效值
    min_valid_disp = 1.0
    filtered_disp[filtered_disp < min_valid_disp] = 0

    # 归一化到0-255以便显示
    disp_normalized = cv2.normalize(filtered_disp, None, alpha=0, beta=255,
                                    norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # 创建伪彩色视差图
    disp_color = cv2.applyColorMap(disp_normalized, cv2.COLORMAP_JET)

    return filtered_disp, disp_normalized, disp_color


def stereoMatchSGBM_optimized(left_image, right_image, config):
    """
    使用SGBM算法计算视差图
    """
    # 确保输入图像是灰度图
    if left_image.ndim != 2:
        left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
    if right_image.ndim != 2:
        right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

    # 创建SGBM匹配器
    left_matcher = cv2.StereoSGBM_create(
        minDisparity=config.minDisparity,
        numDisparities=config.numDisparities,
        blockSize=config.blockSize,
        P1=config.P1,
        P2=config.P2,
        disp12MaxDiff=config.disp12MaxDiff,
        preFilterCap=config.preFilterCap,
        uniquenessRatio=config.uniquenessRatio,
        speckleWindowSize=config.speckleWindowSize,
        speckleRange=config.speckleRange,
        mode=config.mode
    )

    # 计算视差图
    disparity = left_matcher.compute(left_image, right_image)

    # 转换为浮点数
    disparity = disparity.astype(np.float32) / 16.0

    # 应用形态学处理改善质量
    kernel = np.ones((3, 3), np.uint8)
    disparity = cv2.morphologyEx(disparity, cv2.MORPH_CLOSE, kernel)

    # 过滤小视差值
    min_valid_disp = 1.0
    disparity[disparity < min_valid_disp] = 0

    # 归一化到0-255以便显示
    disp_normalized = cv2.normalize(disparity, None, alpha=0, beta=255,
                                    norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # 创建伪彩色视差图
    disp_color = cv2.applyColorMap(disp_normalized, cv2.COLORMAP_JET)

    return disparity, disp_normalized, disp_color


# -------------------------------- 新增深度图处理函数 --------------------------------
def disparity_to_depth(disparity, stereo_config, config=None):
    """
    将视差图转换为深度图(以毫米为单位)

    参数:
    - disparity: 视差图
    - stereo_config: 相机配置对象，包含基线距离和焦距
    - config: 程序配置对象，包含最大/最小距离参数

    返回:
    - depth_map: 深度图，单位为毫米
    """
    # 避免除以零
    valid_mask = disparity > 0

    # 初始化深度图
    depth = np.zeros_like(disparity, dtype=np.float32)

    # 应用公式: 深度 = (基线距离 × 焦距) / 视差
    depth[valid_mask] = (stereo_config.baseline * stereo_config.focal_length) / disparity[valid_mask]

    # 过滤异常值
    if config is not None:
        depth[depth > config.max_distance_mm] = config.max_distance_mm
        depth[depth < config.min_distance_mm] = 0
    else:
        # 默认值
        depth[depth > 10000.0] = 10000.0  # 10米
        depth[depth < 100.0] = 0  # 10厘米

    return depth


def visualize_depth(depth, min_depth, max_depth, colormap=cv2.COLORMAP_INFERNO):
    """
    创建深度图的可视化表示

    参数:
    - depth: 深度图，单位为毫米
    - min_depth: 最小显示深度
    - max_depth: 最大显示深度
    - colormap: 颜色映射方案

    返回:
    - depth_colormap: 彩色深度图
    """
    # 裁剪深度范围
    depth_limited = np.clip(depth, min_depth, max_depth)

    # 归一化到[0,255]范围
    depth_normalized = ((depth_limited - min_depth) / (max_depth - min_depth) * 255).astype(np.uint8)

    # 无效区域(深度为0)设为黑色
    depth_normalized[depth <= 0] = 0

    # 应用颜色映射
    depth_colormap = cv2.applyColorMap(depth_normalized, colormap)

    # 将无效区域设置为黑色
    depth_colormap[depth <= 0] = [0, 0, 0]

    # 添加颜色映射图例
    depth_colormap = add_depth_colorbar(depth_colormap, min_depth, max_depth, colormap)

    return depth_colormap


def add_depth_colorbar(image, min_depth, max_depth, colormap, bar_height=30, margin=10):
    """
    在深度图上添加颜色条图例

    参数:
    - image: 输入图像
    - min_depth: 最小深度值 (毫米)
    - max_depth: 最大深度值 (毫米)
    - colormap: 使用的颜色映射
    - bar_height: 颜色条高度
    - margin: 边距

    返回:
    - 添加了颜色条的图像
    """
    h, w = image.shape[:2]

    # 创建颜色条
    colorbar_width = w - 2 * margin
    colorbar = np.linspace(0, 255, colorbar_width).astype(np.uint8)
    colorbar = np.tile(colorbar, (bar_height, 1))
    colorbar_colored = cv2.applyColorMap(colorbar, colormap)

    # 创建包含颜色条的新图像
    new_height = h + bar_height + 2 * margin + 25  # 额外空间用于文本
    result = np.zeros((new_height, w, 3), dtype=np.uint8)
    result[:h, :] = image

    # 添加颜色条
    result[h + margin:h + margin + bar_height, margin:margin + colorbar_width] = colorbar_colored

    # 添加刻度标签
    num_ticks = 5
    tick_positions = np.linspace(margin, margin + colorbar_width, num_ticks).astype(int)
    tick_values = np.linspace(min_depth, max_depth, num_ticks)

    for pos, val in zip(tick_positions, tick_values):
        # 绘制刻度线
        cv2.line(result, (pos, h + margin + bar_height), (pos, h + margin + bar_height + 5), (255, 255, 255), 1)

        # 添加刻度标签
        label = f"{val / 1000:.1f}m"
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        text_x = pos - text_size[0] // 2
        text_y = h + margin + bar_height + 20
        cv2.putText(result, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # 添加标题
    title = "depth (m)"
    title_size, _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    title_x = w // 2 - title_size[0] // 2
    title_y = h + margin // 2
    cv2.putText(result, title, (title_x, title_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return result


def reprojectTo3D(disparity, Q):
    """
    将视差图转换为3D点云
    """
    # 过滤太小的视差值，这些值会导致点投射到非常远的地方
    filtered_disp = disparity.copy()
    min_disparity = 1.0  # 设置最小视差阈值
    filtered_disp[filtered_disp < min_disparity] = 0

    # 使用OpenCV的reprojectImageTo3D进行重投影
    points_3d = cv2.reprojectImageTo3D(filtered_disp, Q)

    # 过滤深度值异常的点
    max_depth = 10000.0  # 最大深度阈值（毫米）
    mask = (points_3d[:, :, 2] > 0) & (points_3d[:, :, 2] < max_depth)

    # 对异常点设置为无效值
    points_3d[~mask] = [0, 0, 0]

    return points_3d


def DepthColor2Cloud_optimized(points_3d, colors, config):
    """
    将3D点和颜色信息转换为适应Open3D的点云格式，并过滤无效点
    """
    try:
        # 确保两个数组的尺寸匹配
        h, w = points_3d.shape[:2]
        colors = cv2.resize(colors, (w, h)) if colors.shape[:2] != (h, w) else colors

        # 确保colors是彩色图像
        if colors.ndim == 2:
            colors = cv2.cvtColor(colors, cv2.COLOR_GRAY2BGR)

        # 展平3D点和颜色数组以便更高效处理
        X = points_3d[:, :, 0].reshape(-1)
        Y = points_3d[:, :, 1].reshape(-1)
        Z = points_3d[:, :, 2].reshape(-1)
        colors_reshaped = colors.reshape(-1, 3)

        # 设置过滤条件
        min_z = config.min_distance_mm
        max_z = config.max_distance_mm
        max_x = config.max_x_mm
        max_y = config.max_y_mm

        # 应用比例校正
        scale_factor = config.scale_correction
        X *= scale_factor
        Y *= scale_factor
        Z *= scale_factor

        # 创建有效点掩码
        valid_mask = (
                (Z > min_z) & (Z < max_z) &
                (X > -max_x) & (X < max_x) &
                (Y > -max_y) & (Y < max_y) &
                np.isfinite(X) & np.isfinite(Y) & np.isfinite(Z)
        )

        # 排除(0,0,0)点
        zero_mask = np.logical_or(
            np.logical_or(np.abs(X) >= 1e-6, np.abs(Y) >= 1e-6),
            np.abs(Z) >= 1e-6
        )
        valid_mask = valid_mask & zero_mask

        # 计算有效点数量
        valid_count = np.sum(valid_mask)

        # 检查有效点数量
        if valid_count < 10:
            print(f"警告：有效点数量太少（{valid_count}），返回最小点云")
            return np.array([[0, 0, 1000, 255, 0, 0], [0, 0, 2000, 0, 255, 0], [0, 0, 3000, 0, 0, 255]])

        # 分配内存以高效提取有效点
        points = np.zeros((valid_count, 3), dtype=np.float32)
        colors_rgb = np.zeros((valid_count, 3), dtype=np.uint8)

        # 提取有效点
        points[:, 0] = X[valid_mask]
        points[:, 1] = Y[valid_mask]
        points[:, 2] = Z[valid_mask]

        # 提取并转换颜色 (BGR -> RGB)
        valid_colors = colors_reshaped[valid_mask]
        colors_rgb[:, 0] = valid_colors[:, 2]  # R
        colors_rgb[:, 1] = valid_colors[:, 1]  # G
        colors_rgb[:, 2] = valid_colors[:, 0]  # B

        # 组合点和颜色
        pointcloud = np.hstack((points, colors_rgb))

        return pointcloud

    except Exception as e:
        print(f"点云生成错误: {e}")
        import traceback
        traceback.print_exc()
        # 返回一个最小点云，避免可视化错误
        return np.array([[0, 0, 1000, 255, 0, 0], [0, 0, 2000, 0, 255, 0], [0, 0, 3000, 0, 0, 255]])


def view_cloud_open3d(pointcloud, config):
    """
    使用Open3D显示点云（非阻塞）
    """
    if pointcloud.shape[0] < 10:
        print("警告：点云点数太少，不显示")
        return

    # 提取点和颜色
    points = pointcloud[:, :3]
    colors = pointcloud[:, 3:] / 255.0  # 颜色归一化到[0,1]

    # 创建Open3D点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # 在新线程中运行可视化，不阻塞主程序
    def show_point_cloud():
        # 创建可视化窗口
        o3d.visualization.draw_geometries([pcd],
                                          window_name="3D点云",
                                          width=1024,
                                          height=768,
                                          left=50,
                                          top=50)

    thread = threading.Thread(target=show_point_cloud)
    thread.daemon = True  # 设置为守护线程，主程序退出时自动结束
    thread.start()


# -------------------------------- 文本绘制函数 --------------------------------
def cv2_put_chinese_text(img, text, org, fontFace, fontScale, color, thickness=1, lineType=cv2.LINE_8,
                         bottomLeftOrigin=False):
    """
    优化版的在OpenCV图像上绘制中文文本函数
    """
    # 计算对应的PIL字体大小
    base_size = 20  # OpenCV的FONT_HERSHEY_SIMPLEX在fontScale=1时大约20像素
    font_size = int(base_size * fontScale)

    # 获取缓存的字体
    font = _get_font(font_size)

    # 转换颜色格式，从BGR到RGB
    color_rgb = (color[2], color[1], color[0])

    # 计算文本大小
    try:
        # 新版本PIL
        text_bbox = ImageDraw.Draw(Image.new('RGB', (1, 1))).textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
    except:
        # 旧版本PIL
        text_width, text_height = ImageDraw.Draw(Image.new('RGB', (1, 1))).textsize(text, font=font)

    # 添加一些边距
    padding = 5
    text_width += padding * 2
    text_height += padding * 2

    # 调整坐标
    x, y = org
    if bottomLeftOrigin:
        y -= text_height

    # 确保坐标在图像范围内
    x = max(0, min(x, img.shape[1] - text_width))
    y = max(0, min(y, img.shape[0] - text_height))

    # 只处理需要绘制文本的ROI区域
    roi = img[y:y + text_height, x:x + text_width].copy()
    if roi.size == 0:
        return img  # 安全检查

    # 将ROI转换为PIL图像
    roi_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))

    # 创建绘图对象并绘制文本
    draw = ImageDraw.Draw(roi_pil)
    draw.text((padding, padding), text, font=font, fill=color_rgb)

    # 将修改后的ROI转换回OpenCV格式
    roi_result = cv2.cvtColor(np.array(roi_pil), cv2.COLOR_RGB2BGR)

    # 将结果放回原图
    img[y:y + text_height, x:x + text_width] = roi_result

    return img


@lru_cache(maxsize=8)
def _get_font(font_size):
    """
    获取并缓存字体对象，避免重复搜索和加载
    """
    font_paths = []
    if os.name == 'nt':  # Windows
        font_paths.extend([
            'C:/Windows/Fonts/simhei.ttf',  # 黑体
            'C:/Windows/Fonts/simsun.ttc',  # 宋体
            'C:/Windows/Fonts/msyh.ttc',  # 微软雅黑
            'C:/Windows/Fonts/arial.ttf'  # Arial (作为后备)
        ])
    elif sys.platform == 'darwin':  # macOS
        font_paths.extend([
            '/System/Library/Fonts/PingFang.ttc',
            '/Library/Fonts/Arial Unicode.ttf',
            '/System/Library/Fonts/AppleSDGothicNeo.ttc'
        ])
    else:  # Linux
        font_paths.extend([
            '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf',
            '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
            '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
        ])

    # 尝试加载字体
    for path in font_paths:
        try:
            if os.path.exists(path):
                return ImageFont.truetype(path, font_size)
        except Exception:
            continue

    # 如果找不到中文字体，使用默认字体
    return ImageFont.load_default()


# 批量文本渲染函数
def batch_put_chinese_text(img, text_items):
    """
    一次性在图像上绘制多个中文文本，减少格式转换次数

    参数:
    - img: OpenCV图像
    - text_items: 列表，每项为 (text, org, fontFace, fontScale, color, thickness, lineType, bottomLeftOrigin)

    返回: 添加了所有文本的图像
    """
    # 一次性转换为PIL图像
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    for item in text_items:
        if len(item) >= 5:  # 至少需要text, org, fontFace, fontScale, color
            text, org, fontFace, fontScale, color = item[:5]
            thickness = item[5] if len(item) > 5 else 1
            lineType = item[6] if len(item) > 6 else cv2.LINE_8
            bottomLeftOrigin = item[7] if len(item) > 7 else False

            # 计算字体大小并获取字体
            base_size = 20
            font_size = int(base_size * fontScale)
            font = _get_font(font_size)

            # 转换颜色
            color_rgb = (color[2], color[1], color[0])

            # 调整位置
            x, y = org
            if bottomLeftOrigin:
                try:
                    text_bbox = draw.textbbox((0, 0), text, font=font)
                    text_height = text_bbox[3] - text_bbox[1]
                except:
                    _, text_height = draw.textsize(text, font=font)
                y -= text_height

            # 绘制文本
            draw.text((x, y), text, font=font, fill=color_rgb)

    # 转换回OpenCV图像
    result = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return result


# -------------------------------- 距离测量函数 --------------------------------
def measure_distance(points_3d, x, y):
    """
    测量指定像素点到相机的距离
    """
    h, w = points_3d.shape[:2]

    # 检查坐标是否在有效范围内
    if not (0 <= x < w and 0 <= y < h):
        return None

    # 获取点的3D坐标
    point_3d = points_3d[y, x]

    # 检查点的有效性
    if np.all(np.isfinite(point_3d)) and not np.all(point_3d == 0):
        # 计算欧几里得距离
        distance = np.sqrt(np.sum(point_3d ** 2))
        return distance / 1000.0  # 转换为米

    return None


def measure_distance_between_points(points_3d, p1, p2):
    """
    测量两个像素点之间的3D空间距离
    """
    h, w = points_3d.shape[:2]

    # 检查坐标是否在有效范围内
    if not (0 <= p1[0] < w and 0 <= p1[1] < h and 0 <= p2[0] < w and 0 <= p2[1] < h):
        return None

    # 获取两个点的3D坐标
    point1_3d = points_3d[p1[1], p1[0]]
    point2_3d = points_3d[p2[1], p2[0]]

    # 检查点的有效性
    if (np.all(np.isfinite(point1_3d)) and not np.all(point1_3d == 0) and
            np.all(np.isfinite(point2_3d)) and not np.all(point2_3d == 0)):
        # 计算欧几里得距离
        distance = np.sqrt(np.sum((point1_3d - point2_3d) ** 2))
        return distance / 1000.0  # 转换为米

    return None


def draw_measure_points(image, measure_points, points_3d):
    """
    在图像上绘制测量点和距离信息
    """
    result = image.copy()

    # 绘制测量点
    for i, (x, y) in enumerate(measure_points):
        # 计算距离
        distance = measure_distance(points_3d, x, y)

        # 绘制点
        if distance is not None:
            cv2.circle(result, (x, y), 5, (0, 0, 255), -1)
            # 显示点编号和距离
            text = f"{i + 1}: {distance:.2f}m"
            result = cv2_put_chinese_text(result, text, (x + 10, y),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.circle(result, (x, y), 5, (0, 255, 255), -1)
            result = cv2_put_chinese_text(result, f"{i + 1}: 无效", (x + 10, y),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # 如果有两个点，绘制连线并显示距离
    if len(measure_points) >= 2:
        p1 = measure_points[-2]
        p2 = measure_points[-1]

        # 绘制连线
        cv2.line(result, p1, p2, (0, 255, 0), 2)

        # 计算距离
        distance = measure_distance_between_points(points_3d, p1, p2)
        if distance is not None:
            # 计算线段中点
            mid_x = (p1[0] + p2[0]) // 2
            mid_y = (p1[1] + p2[1]) // 2

            # 在中点附近显示距离
            text = f"距离: {distance:.2f}m"
            result = cv2_put_chinese_text(result, text, (mid_x, mid_y - 10),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return result


# -------------------------------- 主程序 --------------------------------
class StereoVisionApp:
    """双目立体视觉应用程序类"""

    def __init__(self):
        """初始化应用程序"""
        self.config = StereoConfig()
        self.running = True

        # 初始化相机
        self.cap = cv2.VideoCapture(self.config.camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)

        # 如果指定相机打开失败，尝试使用默认相机
        if not self.cap.isOpened():
            print(f"无法打开相机ID {self.config.camera_id}，尝试使用默认相机...")
            self.config.camera_id = 1
            self.cap = cv2.VideoCapture(self.config.camera_id)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)

            if not self.cap.isOpened():
                raise RuntimeError("错误：无法打开相机！")

        # 创建窗口
        for name, title in self.config.window_names.items():
            cv2.namedWindow(title, cv2.WINDOW_NORMAL)

        # 设置鼠标回调函数
        cv2.setMouseCallback(self.config.window_names["disparity"], self.mouse_callback)

        # 用于显示点云
        self.point_cloud_enabled = False

        # 双目相机校正参数
        self.stereo_config = stereoCamera()

        # 预先计算校正变换矩阵
        height, width = self.config.frame_height, self.config.frame_width // 2
        self.map1x, self.map1y, self.map2x, self.map2y, self.Q = getRectifyTransform(height, width, self.stereo_config)

        # 打印操作说明
        self.print_instructions()

    def print_instructions(self):
        """显示操作说明"""
        print("\n----- 双目立体视觉系统 -----")
        print("操作说明:")
        print("- 鼠标左键: 在视差图上点击添加测量点")
        print("- 鼠标右键: 清除所有测量点")
        print("- 按 'p' 键: 显示3D点云")
        print("- 按 'c' 键: 保存当前帧")
        print("- 按 'q' 键: 退出程序")
        print("-------------------------\n")

    def mouse_callback(self, event, x, y, flags, param):
        """鼠标事件处理函数"""
        if not self.config.enable_click_measure:
            return

        h, w = self.config.last_disp_size

        # 检查坐标是否在视差图范围内
        if not (0 <= x < w and 0 <= y < h):
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            # 添加测量点
            self.config.measure_points.append((x, y))
            # 如果点超过最大数量，移除最早添加的点
            if len(self.config.measure_points) > self.config.max_measure_points:
                self.config.measure_points.pop(0)
            print(f"添加测量点: ({x}, {y})")

        elif event == cv2.EVENT_RBUTTONDOWN:
            # 清空所有测量点
            self.config.measure_points.clear()
            print("已清除所有测量点")

    def run(self):
        """运行主循环"""
        try:
            while self.running:
                # 读取一帧
                ret, frame = self.cap.read()
                if not ret:
                    print("无法获取图像，尝试重新连接...")
                    time.sleep(0.5)
                    continue

                # 调整图像大小（如果需要）
                if frame.shape[1] != self.config.frame_width or frame.shape[0] != self.config.frame_height:
                    frame = cv2.resize(frame, (self.config.frame_width, self.config.frame_height))

                # 分割左右图像
                mid_x = frame.shape[1] // 2
                left_half = frame[:, :mid_x]
                right_half = frame[:, mid_x:]

                # 消除畸变
                try:
                    iml = undistortion(left_half, self.stereo_config.cam_matrix_left, self.stereo_config.distortion_l)
                    imr = undistortion(right_half, self.stereo_config.cam_matrix_right, self.stereo_config.distortion_r)
                except Exception as e:
                    print(f"畸变校正错误: {e}")
                    continue

                # 预处理图像
                iml_, imr_ = preprocess(iml, imr)

                # 图像校正
                iml_rectified, imr_rectified = rectifyImage(iml_, imr_, self.map1x, self.map1y, self.map2x, self.map2y)

                # 绘制校正线，检查校正效果
                line_image = draw_line(iml_rectified, imr_rectified)

                # 计算视差图 (使用WLS滤波器改善质量)
                disparity, disp_normalized, disp_color = stereoMatchSGBM_WLS(
                    iml_rectified,
                    imr_rectified,
                    self.config
                )

                # 保存视差图尺寸供鼠标回调使用
                self.config.last_disp_size = disp_color.shape[:2]

                # 计算3D点云
                points_3d = reprojectTo3D(disparity, self.Q)

                # 应用比例校正
                if self.config.scale_correction != 1.0:
                    points_3d *= self.config.scale_correction

                # 显示原始左右图像
                original_concat = np.hstack((left_half, right_half))
                cv2.imshow(self.config.window_names["original"], original_concat)

                # 显示校正后的图像
                cv2.imshow(self.config.window_names["rectified"], line_image)

                # 在视差图上绘制测量点
                if self.config.measure_points:
                    disp_with_measures = draw_measure_points(disp_color, self.config.measure_points, points_3d)
                    cv2.imshow(self.config.window_names["disparity"], disp_with_measures)
                else:
                    cv2.imshow(self.config.window_names["disparity"], disp_color)

                # 计算并显示深度图 (新增)
                depth_map = disparity_to_depth(disparity, self.stereo_config)
                depth_color = visualize_depth(
                    depth_map,
                    self.config.depth_min_display,
                    self.config.depth_max_display,
                    self.config.depth_colormap
                )
                cv2.imshow(self.config.window_names["depth"], depth_color)

                # 显示点云（如果启用）
                if self.point_cloud_enabled and points_3d is not None:
                    try:
                        # 生成点云
                        pointcloud = DepthColor2Cloud_optimized(points_3d, left_half, self.config)
                        print(f"生成点云成功，点数: {pointcloud.shape[0]}")

                        # 显示点云
                        view_cloud_open3d(pointcloud, self.config)
                        self.point_cloud_enabled = False  # 重置标志
                    except Exception as e:
                        print(f"点云生成错误: {e}")

                # 键盘控制
                key = cv2.waitKey(1) & 0xFF
                self.handle_keyboard(key)

                # 限制帧率，防止CPU占用过高
                time.sleep(1.0 / self.config.fps_limit)

        finally:
            # 清理资源
            self.cleanup()

    def handle_keyboard(self, key):
        """处理键盘输入"""
        if key == ord('q'):  # 按'q'退出
            self.running = False

        elif key == ord('p'):  # 按'p'显示点云
            self.point_cloud_enabled = True
            print("正在生成点云...")

        elif key == ord('c'):  # 按'c'保存当前帧
            self.save_current_frame()

    def save_current_frame(self):
        """保存当前帧"""
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")

            # 创建输出目录
            os.makedirs(self.config.output_dir, exist_ok=True)

            # 读取一帧
            ret, frame = self.cap.read()
            if not ret:
                print("无法获取图像，无法保存")
                return

            # 分割左右图像
            mid_x = frame.shape[1] // 2
            left_half = frame[:, :mid_x]
            right_half = frame[:, mid_x:]

            # 保存图像
            left_filename = os.path.join(self.config.output_dir, f"left_{timestamp}.jpg")
            right_filename = os.path.join(self.config.output_dir, f"right_{timestamp}.jpg")

            cv2.imwrite(left_filename, left_half)
            cv2.imwrite(right_filename, right_half)

            print(f"已保存图像到: {left_filename}, {right_filename}")

        except Exception as e:
            print(f"保存图像失败: {e}")

    def cleanup(self):
        """清理资源"""
        print("正在关闭程序...")

        # 释放相机
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()

        # 关闭所有窗口
        cv2.destroyAllWindows()
        print("程序已退出")


# 主函数
if __name__ == "__main__":
    try:
        app = StereoVisionApp()
        app.run()
    except Exception as e:
        print(f"程序发生错误: {e}")
        import traceback

        traceback.print_exc()