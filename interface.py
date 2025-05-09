#!/usr/bin/env python
# -*- coding: utf-8 -*-



import sys
import os
import logging
import traceback
import numpy as np
import pandas as pd
from PySide6.QtCore import Qt, Signal, Slot, QTimer, QPointF, QRectF, QSize, QEvent, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import (
    QColor, QFont, QAction, QPainter, QPen, QBrush, QPainterPath, QIcon, QFontDatabase,
    QLinearGradient, QPixmap, QPalette, QImage, QCursor, QKeySequence, QTransform
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QPushButton, QFileDialog, QTableWidget,
    QTableWidgetItem, QProgressBar, QMessageBox, QComboBox,
    QSpinBox, QGroupBox, QGridLayout, QSplitter, QLineEdit,
    QTextEdit, QHeaderView, QDoubleSpinBox, QFrame, QToolBar,
    QToolButton, QMenu, QSizePolicy, QScrollArea, QStatusBar,
    QDialog, QCheckBox, QSlider, QButtonGroup, QRadioButton,
    QListWidget, QListWidgetItem, QStackedWidget, QGraphicsDropShadowEffect,
    QStyle, QStyleFactory, QStyledItemDelegate, QToolTip
)

# 导入推荐系统核心类
try:
    from library_recommender import LibraryRecommender, setup_logging, error_handler, create_demo_data
except ImportError:
    # 错误提示，帮助用户理解问题
    print("错误: 无法导入 library_recommender 模块")
    print("请确保将原始推荐系统代码保存为 library_recommender.py 并与此文件放置在同一目录")
    sys.exit(1)


class HybridRecommendationWidget(QWidget):
    """
    混合推荐窗口
    """

    def __init__(self, recommender, theme, icons, parent=None):
        """
        初始化窗口

        参数:
            recommender: LibraryRecommender实例
            theme: 主题对象
            icons: 图标字典
            parent: 父窗口
        """
        super(HybridRecommendationWidget, self).__init__(parent)
        self.recommender = recommender
        self.theme = theme
        self.icons = icons
        self.recommendations = None
        self.optimal_weights = None
        self.init_ui()

    def init_ui(self):
        """
        初始化UI组件
        """
        # 创建主布局
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(20)

        # 顶部操作栏
        top_bar = QHBoxLayout()

        # 标题
        title_label = QLabel("混合推荐系统")
        font = title_label.font()
        font.setPointSize(16)
        font.setBold(True)
        title_label.setFont(font)
        top_bar.addWidget(title_label)

        top_bar.addStretch()

        # 优化按钮
        self.optimize_btn = VSCodeButton("自动优化权重", self.icons['evaluate'])
        self.optimize_btn.clicked.connect(self.optimize_weights)
        top_bar.addWidget(self.optimize_btn)

        main_layout.addLayout(top_bar)

        # 上半部分：控制面板和权重可视化
        top_section = QSplitter(Qt.Horizontal)

        # 控制面板
        control_card = CardWidget("推荐参数设置")
        control_layout = QVBoxLayout()

        # 参数设置
        params_group = QWidget()
        params_layout = QGridLayout(params_group)
        params_layout.setColumnStretch(1, 1)  # 设置第二列可伸展

        # 用户选择
        self.user_label = QLabel("选择用户:")
        self.user_combo = QComboBox()

        # 推荐数量
        self.recommendations_label = QLabel("推荐数量:")
        self.recommendations_spin = QSpinBox()
        self.recommendations_spin.setRange(1, 20)
        self.recommendations_spin.setValue(10)
        self.recommendations_spin.setToolTip("要生成的推荐图书数量")

        # 添加到布局
        params_layout.addWidget(self.user_label, 0, 0)
        params_layout.addWidget(self.user_combo, 0, 1)

        params_layout.addWidget(self.recommendations_label, 1, 0)
        params_layout.addWidget(self.recommendations_spin, 1, 1)

        control_layout.addWidget(params_group)

        # 权重设置
        weights_group = QGroupBox("推荐方法权重")
        weights_layout = QVBoxLayout(weights_group)

        # 基于用户的推荐权重
        user_weight_layout = QHBoxLayout()
        self.user_weight_label = QLabel("基于用户的推荐:")
        self.user_weight_slider = QSlider(Qt.Horizontal)
        self.user_weight_slider.setRange(0, 100)
        self.user_weight_slider.setValue(33)
        self.user_weight_slider.setTickPosition(QSlider.TicksBelow)
        self.user_weight_slider.setTickInterval(10)
        self.user_weight_slider.valueChanged.connect(self.update_weights)
        self.user_weight_value = QLabel("0.33")

        user_weight_layout.addWidget(self.user_weight_label)
        user_weight_layout.addWidget(self.user_weight_slider)
        user_weight_layout.addWidget(self.user_weight_value)
        weights_layout.addLayout(user_weight_layout)

        # 基于物品的推荐权重
        item_weight_layout = QHBoxLayout()
        self.item_weight_label = QLabel("基于物品的推荐:")
        self.item_weight_slider = QSlider(Qt.Horizontal)
        self.item_weight_slider.setRange(0, 100)
        self.item_weight_slider.setValue(33)
        self.item_weight_slider.setTickPosition(QSlider.TicksBelow)
        self.item_weight_slider.setTickInterval(10)
        self.item_weight_slider.valueChanged.connect(self.update_weights)
        self.item_weight_value = QLabel("0.33")

        item_weight_layout.addWidget(self.item_weight_label)
        item_weight_layout.addWidget(self.item_weight_slider)
        item_weight_layout.addWidget(self.item_weight_value)
        weights_layout.addLayout(item_weight_layout)

        # 矩阵分解推荐权重
        matrix_weight_layout = QHBoxLayout()
        self.matrix_weight_label = QLabel("矩阵分解推荐:")
        self.matrix_weight_slider = QSlider(Qt.Horizontal)
        self.matrix_weight_slider.setRange(0, 100)
        self.matrix_weight_slider.setValue(34)
        self.matrix_weight_slider.setTickPosition(QSlider.TicksBelow)
        self.matrix_weight_slider.setTickInterval(10)
        self.matrix_weight_slider.valueChanged.connect(self.update_weights)
        self.matrix_weight_value = QLabel("0.34")

        matrix_weight_layout.addWidget(self.matrix_weight_label)
        matrix_weight_layout.addWidget(self.matrix_weight_slider)
        matrix_weight_layout.addWidget(self.matrix_weight_value)
        weights_layout.addLayout(matrix_weight_layout)

        control_layout.addWidget(weights_group)

        # 添加分隔线
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        control_layout.addWidget(separator)

        # 选择使用的方法
        methods_group = QGroupBox("使用的推荐方法")
        methods_layout = QVBoxLayout(methods_group)

        self.use_user_based = QCheckBox("使用基于用户的推荐")
        self.use_user_based.setChecked(True)
        self.use_user_based.stateChanged.connect(self.update_method_selection)
        methods_layout.addWidget(self.use_user_based)

        self.use_item_based = QCheckBox("使用基于物品的推荐")
        self.use_item_based.setChecked(True)
        self.use_item_based.stateChanged.connect(self.update_method_selection)
        methods_layout.addWidget(self.use_item_based)

        self.use_matrix_factorization = QCheckBox("使用矩阵分解推荐")
        self.use_matrix_factorization.setChecked(True)
        self.use_matrix_factorization.stateChanged.connect(self.update_method_selection)
        methods_layout.addWidget(self.use_matrix_factorization)

        control_layout.addWidget(methods_group)

        # 操作按钮区
        buttons_layout = QVBoxLayout()

        self.recommend_btn = VSCodeButton("生成混合推荐", self.icons['book'])
        self.recommend_btn.clicked.connect(self.generate_recommendations)
        buttons_layout.addWidget(self.recommend_btn)

        self.save_btn = VSCodeButton("保存推荐结果", None)
        self.save_btn.clicked.connect(self.save_recommendations)
        self.save_btn.setEnabled(False)
        buttons_layout.addWidget(self.save_btn)

        control_layout.addLayout(buttons_layout)

        # 添加伸缩器
        control_layout.addStretch()

        control_card.add_layout(control_layout)
        top_section.addWidget(control_card)

        # 权重可视化
        weights_card = CardWidget("权重可视化")
        weights_layout = QVBoxLayout()

        self.weights_chart = ModernChartWidget()
        self.weights_chart.set_title("推荐方法权重")
        self.weights_chart.set_labels("推荐方法", "权重")
        self.update_weights_chart()

        weights_layout.addWidget(self.weights_chart)

        weights_card.add_layout(weights_layout)
        top_section.addWidget(weights_card)

        # 设置分割比例
        top_section.setSizes([400, 600])

        main_layout.addWidget(top_section, 1)  # 1是伸展因子

        # 下半部分：推荐过程和结果
        bottom_section = QSplitter(Qt.Horizontal)

        # 推荐过程显示
        process_card = CardWidget("推荐过程")
        process_layout = QVBoxLayout()

        self.process_text = QTextEdit()
        self.process_text.setReadOnly(True)

        process_layout.addWidget(self.process_text)

        process_card.add_layout(process_layout)
        bottom_section.addWidget(process_card)

        # 推荐结果
        results_card = CardWidget("推荐结果")
        results_layout = QVBoxLayout()

        self.results_table = QTableWidget()
        self.results_table.setColumnCount(3)
        self.results_table.setHorizontalHeaderLabels(["图书ID", "图书标题", "预测评分"])
        self.results_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)

        results_layout.addWidget(self.results_table)

        results_card.add_layout(results_layout)
        bottom_section.addWidget(results_card)

        # 设置分割比例
        bottom_section.setSizes([500, 500])

        main_layout.addWidget(bottom_section, 1)  # 1是伸展因子

    def update_user_list(self):
        """
        更新用户下拉列表
        """
        try:
            self.user_combo.clear()

            if self.recommender.users_df is not None:
                for user_id in sorted(self.recommender.users_df['user_id'].unique()):
                    self.user_combo.addItem(str(user_id))
        except Exception as e:
            logging.error(f"更新用户列表时发生错误: {str(e)}")
            QMessageBox.critical(self, "错误", f"更新用户列表时发生错误:\n{str(e)}")

    def update_weights(self):
        """
        更新权重显示
        """
        try:
            # 获取各滑块的值
            user_value = self.user_weight_slider.value()
            item_value = self.item_weight_slider.value()
            matrix_value = self.matrix_weight_slider.value()

            # 归一化权重
            total = user_value + item_value + matrix_value
            if total > 0:
                user_weight = user_value / total
                item_weight = item_value / total
                matrix_weight = matrix_value / total
            else:
                user_weight = item_weight = matrix_weight = 1 / 3

            # 更新标签显示
            self.user_weight_value.setText(f"{user_weight:.2f}")
            self.item_weight_value.setText(f"{item_weight:.2f}")
            self.matrix_weight_value.setText(f"{matrix_weight:.2f}")

            # 更新图表
            self.update_weights_chart()

        except Exception as e:
            logging.error(f"更新权重显示时发生错误: {str(e)}")

    def update_weights_chart(self):
        """
        更新权重可视化图表
        """
        try:
            # 获取所选方法和权重
            methods, weights = self.get_selected_methods_and_weights()

            # 方法名称映射
            method_names = {
                'user_based': '基于用户',
                'item_based': '基于物品',
                'matrix_factorization': '矩阵分解'
            }

            # 准备图表数据
            labels = [method_names.get(method, method) for method in methods]

            # 更新图表
            self.weights_chart.set_bar_data(labels, weights)

        except Exception as e:
            logging.error(f"更新权重图表时发生错误: {str(e)}")

    def update_method_selection(self):
        """
        更新选中的推荐方法
        """
        try:
            # 至少需要选择一种方法
            if not (self.use_user_based.isChecked() or
                    self.use_item_based.isChecked() or
                    self.use_matrix_factorization.isChecked()):
                # 如果全部取消选中，则重新选中发生变化的那个
                sender = self.sender()
                if sender:
                    sender.setChecked(True)
                    QMessageBox.warning(self, "警告", "至少需要选择一种推荐方法")

            # 更新图表
            self.update_weights_chart()

        except Exception as e:
            logging.error(f"更新方法选择时发生错误: {str(e)}")

    def get_selected_methods_and_weights(self):
        """
        获取选中的方法和对应权重

        返回:
            tuple: (methods, weights) 方法列表和对应权重
        """
        methods = []
        raw_weights = []

        # 收集选中的方法
        if self.use_user_based.isChecked():
            methods.append('user_based')
            raw_weights.append(self.user_weight_slider.value())

        if self.use_item_based.isChecked():
            methods.append('item_based')
            raw_weights.append(self.item_weight_slider.value())

        if self.use_matrix_factorization.isChecked():
            methods.append('matrix_factorization')
            raw_weights.append(self.matrix_weight_slider.value())

        # 归一化权重
        total = sum(raw_weights)
        if total > 0:
            weights = [w / total for w in raw_weights]
        else:
            weights = [1.0 / len(methods)] * len(methods)

        return methods, weights

    @error_handler
    def generate_recommendations(self):
        """
        生成混合推荐
        """
        try:
            # 获取参数
            user_id = int(self.user_combo.currentText())
            n_recommendations = self.recommendations_spin.value()
            methods, weights = self.get_selected_methods_and_weights()

            # 清空之前的结果
            self.results_table.setRowCount(0)

            # 添加过程开始信息
            self.process_text.setHtml(f"""
            <style>
            h3, h4 {{
                color: #0078D4;
                margin-top: 16px;
                margin-bottom: 8px;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin-bottom: 16px;
            }}
            th, td {{
                padding: 8px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: rgba(0, 122, 204, 0.2);
            }}
            hr {{
                border: none;
                border-top: 1px solid #ddd;
                margin: 16px 0;
            }}
            </style>

            <h3>为用户 {user_id} 生成混合推荐</h3>
            <p>使用以下方法和权重:</p>
            <table>
                <tr>
                    <th>推荐方法</th>
                    <th>权重</th>
                </tr>
            """)

            # 方法名称映射
            method_names = {
                'user_based': '基于用户的协同过滤',
                'item_based': '基于物品的协同过滤',
                'matrix_factorization': '矩阵分解'
            }

            for method, weight in zip(methods, weights):
                self.process_text.insertHtml(f"""
                <tr>
                    <td>{method_names.get(method, method)}</td>
                    <td>{weight:.2f}</td>
                </tr>
                """)

            self.process_text.insertHtml("""
            </table>
            <hr>
            """)

            # 检查是否需要先训练矩阵分解模型
            if 'matrix_factorization' in methods:
                if not hasattr(self.recommender, 'user_factors') or self.recommender.user_factors is None:
                    self.process_text.insertHtml("<p>矩阵分解模型尚未训练，正在训练模型...</p>")
                    QApplication.processEvents()

                    # 训练矩阵分解模型
                    self.recommender.train_matrix_factorization()

                    self.process_text.insertHtml("<p>矩阵分解模型训练完成</p>")
                    QApplication.processEvents()

            # 生成推荐
            self.process_text.insertHtml("<h4>正在生成混合推荐...</h4>")
            QApplication.processEvents()

            self.recommendations = self.recommender.get_hybrid_recommendations(
                user_id=user_id,
                n_recommendations=n_recommendations,
                methods=methods,
                weights=weights
            )

            # 显示推荐结果
            self.display_recommendations()

            # 使用混合推荐的原理说明
            self.process_text.insertHtml("""
            <h4>混合推荐原理</h4>
            <p>混合推荐系统结合了多种推荐算法的优势，主要步骤如下:</p>
            <ol>
                <li>分别使用各个推荐算法生成候选推荐列表</li>
                <li>对每种算法的推荐结果按设定的权重进行加权</li>
                <li>合并所有推荐结果，对于同一本书出现在多个算法的推荐中，将其加权分数相加</li>
                <li>按合并后的分数排序，选择得分最高的图书作为最终推荐</li>
            </ol>
            <p>这种混合方法通常能够克服单一算法的局限性，提供更全面和准确的推荐。</p>
            <hr>
            """)

            # 启用保存按钮
            self.save_btn.setEnabled(True)

            # 提示成功
            self.process_text.insertHtml("<h4>混合推荐生成完成</h4>")

        except Exception as e:
            logging.error(f"生成混合推荐时发生错误: {str(e)}")
            QMessageBox.critical(self, "错误", f"生成混合推荐时发生错误:\n{str(e)}")

    @error_handler
    def display_recommendations(self):
        """
        在表格中显示推荐结果
        """
        try:
            if not self.recommendations:
                return

            # 清空表格
            self.results_table.setRowCount(0)

            # 添加推荐结果
            for i, (book_id, book_title, score) in enumerate(self.recommendations):
                self.results_table.insertRow(i)

                # 添加图书ID
                id_item = QTableWidgetItem(str(book_id))
                id_item.setTextAlignment(Qt.AlignCenter)
                self.results_table.setItem(i, 0, id_item)

                # 添加图书标题
                title_item = QTableWidgetItem(book_title)
                self.results_table.setItem(i, 1, title_item)

                # 添加预测评分
                score_item = QTableWidgetItem(f"{score:.2f}")
                score_item.setTextAlignment(Qt.AlignCenter)

                # 根据评分设置背景颜色
                if score >= 4.5:
                    score_item.setBackground(QColor(108, 198, 68, 100))  # 绿色
                elif score >= 4.0:
                    score_item.setBackground(QColor(0, 122, 204, 100))  # 蓝色
                elif score >= 3.5:
                    score_item.setBackground(QColor(225, 151, 76, 100))  # 橙色

                self.results_table.setItem(i, 2, score_item)

            # 调整列宽
            self.results_table.resizeColumnsToContents()

        except Exception as e:
            logging.error(f"显示推荐结果时发生错误: {str(e)}")

    @error_handler
    def optimize_weights(self):
        """
        自动优化混合推荐的权重
        """
        try:
            # 获取当前选中的方法
            methods, _ = self.get_selected_methods_and_weights()

            if not methods:
                QMessageBox.warning(self, "警告", "请至少选择一种推荐方法")
                return

            # 显示进度对话框
            progress_dialog = QDialog(self)
            progress_dialog.setWindowTitle("正在优化")
            progress_dialog.setFixedSize(350, 150)

            dialog_layout = QVBoxLayout(progress_dialog)

            info_label = QLabel("正在优化混合推荐权重，这可能需要几分钟时间...", progress_dialog)
            info_label.setWordWrap(True)
            dialog_layout.addWidget(info_label)

            # 添加详细说明
            detail_label = QLabel(
                "正在使用交叉验证和随机搜索找到最优权重。"
                "系统将测试多种权重组合并选择评估效果最好的一组。",
                progress_dialog
            )
            detail_label.setWordWrap(True)
            dialog_layout.addWidget(detail_label)

            progress_bar = QProgressBar(progress_dialog)
            progress_bar.setRange(0, 0)  # 设置为不确定模式
            dialog_layout.addWidget(progress_bar)

            progress_dialog.show()
            QApplication.processEvents()

            # 准备用于优化的测试数据
            ratings = self.recommender.ratings_df
            np.random.seed(42)
            test_indices = np.random.choice(ratings.index, size=int(len(ratings) * 0.2), replace=False)
            test_data = ratings.loc[test_indices]

            # 优化权重
            result = self.recommender.optimize_hybrid_weights(
                test_data=test_data,
                methods=methods,
                n_fold=3,  # 减少折数以加速优化
                n_trials=10  # 减少试验次数以加速优化
            )

            # 关闭进度对话框
            progress_dialog.close()

            if result and 'weights' in result:
                self.optimal_weights = result

                # 更新UI上的权重滑块
                methods = result['methods']
                weights = result['weights']

                # 重置滑块值
                self.user_weight_slider.setValue(0)
                self.item_weight_slider.setValue(0)
                self.matrix_weight_slider.setValue(0)

                # 将优化权重转换为滑块值
                slider_values = [int(w * 100) for w in weights]

                # 设置新的滑块值
                for method, value in zip(methods, slider_values):
                    if method == 'user_based':
                        self.user_weight_slider.setValue(value)
                    elif method == 'item_based':
                        self.item_weight_slider.setValue(value)
                    elif method == 'matrix_factorization':
                        self.matrix_weight_slider.setValue(value)

                # 显示优化结果
                rmse = result.get('rmse', 'N/A')
                methods_str = ', '.join(methods)
                weights_str = ', '.join([f"{w:.2f}" for w in weights])

                self.process_text.setHtml(f"""
                <style>
                h3, h4 {{
                    color: #0078D4;
                    margin-top: 16px;
                    margin-bottom: 8px;
                }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                    margin-bottom: 16px;
                }}
                th, td {{
                    padding: 8px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }}
                th {{
                    background-color: rgba(0, 122, 204, 0.2);
                }}
                hr {{
                    border: none;
                    border-top: 1px solid #ddd;
                    margin: 16px 0;
                }}
                .highlight {{
                    background-color: rgba(0, 122, 204, 0.1);
                    padding: 15px;
                    border-radius: 5px;
                }}
                </style>

                <h3>权重优化结果</h3>
                <div class="highlight">
                    <p><b>推荐方法:</b> {methods_str}</p>
                    <p><b>最优权重:</b> {weights_str}</p>
                    <p><b>RMSE:</b> {rmse if isinstance(rmse, (int, float)) else rmse}</p>
                </div>
                <p>这些权重已经应用到权重滑块上，您可以直接生成推荐或进行微调。</p>
                <hr>
                """)

                # 消息提示
                QMessageBox.information(self, "优化完成", f"已找到最优权重组合，RMSE: {rmse}")
            else:
                QMessageBox.warning(self, "优化结果", "未能找到有效的最优权重组合")

        except Exception as e:
            logging.error(f"优化混合推荐权重时发生错误: {str(e)}")
            QMessageBox.critical(self, "错误", f"优化混合推荐权重时发生错误:\n{str(e)}")

    @error_handler
    def save_recommendations(self):
        """
        保存推荐结果到CSV文件
        """
        try:
            if not self.recommendations:
                QMessageBox.warning(self, "警告", "没有推荐结果可保存")
                return

            # 选择保存文件
            file_path, _ = QFileDialog.getSaveFileName(
                self, "保存推荐结果", "", "CSV文件 (*.csv)"
            )

            if not file_path:
                return

            # 保存推荐结果
            self.recommender.save_recommendations(self.recommendations, file_path)

            # 提示成功
            QMessageBox.information(self, "成功", f"推荐结果已保存到:\n{file_path}")

        except Exception as e:
            logging.error(f"保存推荐结果时发生错误: {str(e)}")
            QMessageBox.critical(self, "错误", f"保存推荐结果时发生错误:\n{str(e)}")


class MatrixFactorizationRecommendationWidget(QWidget):
    """
    基于矩阵分解的推荐窗口
    """

    def __init__(self, recommender, theme, icons, parent=None):
        """
        初始化窗口

        参数:
            recommender: LibraryRecommender实例
            theme: 主题对象
            icons: 图标字典
            parent: 父窗口
        """
        super(MatrixFactorizationRecommendationWidget, self).__init__(parent)
        self.recommender = recommender
        self.theme = theme
        self.icons = icons
        self.recommendations = None
        self.init_ui()

    def init_ui(self):
        """
        初始化UI组件
        """
        # 创建主布局
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(20)

        # 顶部操作栏
        top_bar = QHBoxLayout()

        # 标题
        title_label = QLabel("基于矩阵分解的推荐")
        font = title_label.font()
        font.setPointSize(16)
        font.setBold(True)
        title_label.setFont(font)
        top_bar.addWidget(title_label)

        top_bar.addStretch()

        # 训练按钮
        self.train_btn = VSCodeButton("训练模型", self.icons['book'])
        self.train_btn.clicked.connect(self.train_model)
        top_bar.addWidget(self.train_btn)

        main_layout.addLayout(top_bar)

        # 上半部分：控制面板和模型可视化
        top_section = QSplitter(Qt.Horizontal)

        # 控制面板
        control_card = CardWidget("模型参数设置")
        control_layout = QVBoxLayout()

        # 参数设置
        params_group = QWidget()
        params_layout = QGridLayout(params_group)
        params_layout.setColumnStretch(1, 1)  # 设置第二列可伸展

        # 用户选择
        self.user_label = QLabel("选择用户:")
        self.user_combo = QComboBox()

        # 模型参数
        self.factors_label = QLabel("潜在因子数量:")
        self.factors_spin = QSpinBox()
        self.factors_spin.setRange(10, 100)
        self.factors_spin.setValue(20)
        self.factors_spin.setToolTip("潜在因子数量影响模型的表达能力")

        self.learning_rate_label = QLabel("学习率:")
        self.learning_rate_spin = QDoubleSpinBox()
        self.learning_rate_spin.setRange(0.0001, 0.1)
        self.learning_rate_spin.setSingleStep(0.001)
        self.learning_rate_spin.setValue(0.005)
        self.learning_rate_spin.setDecimals(4)
        self.learning_rate_spin.setToolTip("梯度下降的学习率")

        self.regularization_label = QLabel("正则化参数:")
        self.regularization_spin = QDoubleSpinBox()
        self.regularization_spin.setRange(0.001, 0.1)
        self.regularization_spin.setSingleStep(0.001)
        self.regularization_spin.setValue(0.02)
        self.regularization_spin.setDecimals(3)
        self.regularization_spin.setToolTip("防止过拟合的正则化参数")

        self.epochs_label = QLabel("训练轮数:")
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(10, 200)
        self.epochs_spin.setValue(50)
        self.epochs_spin.setToolTip("训练迭代次数")

        self.recommendations_label = QLabel("推荐数量:")
        self.recommendations_spin = QSpinBox()
        self.recommendations_spin.setRange(1, 20)
        self.recommendations_spin.setValue(10)
        self.recommendations_spin.setToolTip("要生成的推荐图书数量")

        # 添加到布局
        params_layout.addWidget(self.user_label, 0, 0)
        params_layout.addWidget(self.user_combo, 0, 1)

        params_layout.addWidget(self.factors_label, 1, 0)
        params_layout.addWidget(self.factors_spin, 1, 1)

        params_layout.addWidget(self.learning_rate_label, 2, 0)
        params_layout.addWidget(self.learning_rate_spin, 2, 1)

        params_layout.addWidget(self.regularization_label, 3, 0)
        params_layout.addWidget(self.regularization_spin, 3, 1)

        params_layout.addWidget(self.epochs_label, 4, 0)
        params_layout.addWidget(self.epochs_spin, 4, 1)

        params_layout.addWidget(self.recommendations_label, 5, 0)
        params_layout.addWidget(self.recommendations_spin, 5, 1)

        control_layout.addWidget(params_group)

        # 添加分隔线
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        control_layout.addWidget(separator)

        # 操作按钮区
        buttons_layout = QVBoxLayout()

        self.recommend_btn = VSCodeButton("生成推荐", self.icons['book'])
        self.recommend_btn.clicked.connect(self.generate_recommendations)
        self.recommend_btn.setEnabled(False)
        buttons_layout.addWidget(self.recommend_btn)

        self.save_btn = VSCodeButton("保存推荐结果", None)
        self.save_btn.clicked.connect(self.save_recommendations)
        self.save_btn.setEnabled(False)
        buttons_layout.addWidget(self.save_btn)

        control_layout.addLayout(buttons_layout)

        # 添加伸缩器
        control_layout.addStretch()

        control_card.add_layout(control_layout)
        top_section.addWidget(control_card)

        # 模型可视化
        visualization_card = CardWidget("模型可视化")
        visualization_layout = QVBoxLayout()

        self.model_chart = ModernChartWidget()
        visualization_layout.addWidget(self.model_chart)

        visualization_card.add_layout(visualization_layout)
        top_section.addWidget(visualization_card)

        # 设置分割比例
        top_section.setSizes([300, 700])

        main_layout.addWidget(top_section, 1)  # 1是伸展因子

        # 下半部分：训练过程可视化和结果
        bottom_section = QSplitter(Qt.Horizontal)

        # 训练过程可视化
        process_card = CardWidget("训练和推荐过程")
        process_layout = QVBoxLayout()

        self.process_text = QTextEdit()
        self.process_text.setReadOnly(True)

        process_layout.addWidget(self.process_text)

        process_card.add_layout(process_layout)
        bottom_section.addWidget(process_card)

        # 推荐结果
        results_card = CardWidget("推荐结果")
        results_layout = QVBoxLayout()

        self.results_table = QTableWidget()
        self.results_table.setColumnCount(3)
        self.results_table.setHorizontalHeaderLabels(["图书ID", "图书标题", "预测评分"])
        self.results_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)

        results_layout.addWidget(self.results_table)

        results_card.add_layout(results_layout)
        bottom_section.addWidget(results_card)

        # 设置分割比例
        bottom_section.setSizes([500, 500])

        main_layout.addWidget(bottom_section, 1)  # 1是伸展因子

    def update_user_list(self):
        """
        更新用户下拉列表
        """
        try:
            self.user_combo.clear()

            if self.recommender.users_df is not None:
                for user_id in sorted(self.recommender.users_df['user_id'].unique()):
                    self.user_combo.addItem(str(user_id))
        except Exception as e:
            logging.error(f"更新用户列表时发生错误: {str(e)}")
            QMessageBox.critical(self, "错误", f"更新用户列表时发生错误:\n{str(e)}")

    @error_handler
    def train_model(self):
        """
        训练矩阵分解模型
        """
        try:
            # 获取参数
            n_factors = self.factors_spin.value()
            learning_rate = self.learning_rate_spin.value()
            regularization = self.regularization_spin.value()
            n_epochs = self.epochs_spin.value()

            # 清空之前的结果
            self.process_text.clear()

            # 添加过程开始信息
            self.process_text.setHtml(f"""
            <style>
            h3, h4 {{
                color: #0078D4;
                margin-top: 16px;
                margin-bottom: 8px;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin-bottom: 16px;
            }}
            th, td {{
                padding: 8px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: rgba(0, 122, 204, 0.2);
            }}
            hr {{
                border: none;
                border-top: 1px solid #ddd;
                margin: 16px 0;
            }}
            .formula {{
                background-color: rgba(0, 122, 204, 0.1);
                padding: 10px;
                border-radius: 4px;
                font-family: monospace;
                text-align: center;
                margin: 10px 0;
            }}
            </style>

            <h3>训练矩阵分解模型</h3>
            <p>模型参数:</p>
            <ul>
                <li>潜在因子数量: {n_factors}</li>
                <li>学习率: {learning_rate}</li>
                <li>正则化参数: {regularization}</li>
                <li>训练轮数: {n_epochs}</li>
            </ul>
            <hr>
            <h4>矩阵分解原理</h4>
            <p>矩阵分解将用户-物品评分矩阵 R 分解为两个低维矩阵的乘积:</p>
            <div class="formula">
                R ≈ P × Q<sup>T</sup>
            </div>
            <p>其中:</p>
            <ul>
                <li>P: 用户特征矩阵</li>
                <li>Q: 物品特征矩阵</li>
            </ul>
            <p>这些矩阵中的行表示用户和物品在潜在因子空间中的向量表示。</p>
            <hr>
            <h4>开始训练...</h4>
            """)
            QApplication.processEvents()

            # 显示进度对话框
            progress_dialog = QDialog(self)
            progress_dialog.setWindowTitle("训练中")
            progress_dialog.setFixedSize(300, 100)

            dialog_layout = QVBoxLayout(progress_dialog)

            info_label = QLabel("正在训练矩阵分解模型...", progress_dialog)
            dialog_layout.addWidget(info_label)

            progress_bar = QProgressBar(progress_dialog)
            progress_bar.setRange(0, n_epochs)
            dialog_layout.addWidget(progress_bar)

            progress_dialog.show()
            QApplication.processEvents()

            # 创建自定义日志处理器来捕获训练进度
            class ProgressLogger(logging.Handler):
                def __init__(self, text_edit, progress_bar):
                    super().__init__()
                    self.text_edit = text_edit
                    self.progress_bar = progress_bar
                    self.current_epoch = 0

                def emit(self, record):
                    msg = self.format(record)
                    if "轮次" in msg:
                        try:
                            epoch = int(msg.split("/")[0].split(" ")[-1])
                            self.current_epoch = epoch
                            self.progress_bar.setValue(epoch)
                            self.text_edit.insertHtml(f"<p>{msg}</p>")
                            self.text_edit.ensureCursorVisible()
                            QApplication.processEvents()
                        except:
                            pass

            # 添加自定义日志处理器
            logger = logging.getLogger()
            progress_handler = ProgressLogger(self.process_text, progress_bar)
            progress_handler.setLevel(logging.INFO)
            logger.addHandler(progress_handler)

            # 训练模型
            self.recommender.train_matrix_factorization(
                n_factors=n_factors,
                learning_rate=learning_rate,
                regularization=regularization,
                n_epochs=n_epochs
            )

            # 移除自定义日志处理器
            logger.removeHandler(progress_handler)

            # 关闭进度对话框
            progress_dialog.close()

            # 显示成功信息
            self.process_text.insertHtml("""
            <h4>训练完成</h4>
            <p>矩阵分解模型已成功训练。您现在可以生成推荐。</p>
            """)

            # 启用推荐按钮
            self.recommend_btn.setEnabled(True)

            # 显示潜在因子可视化
            self.visualize_latent_factors()

            # 提示成功
            QMessageBox.information(self, "成功", "矩阵分解模型训练完成！")

        except Exception as e:
            logging.error(f"训练矩阵分解模型时发生错误: {str(e)}")
            QMessageBox.critical(self, "错误", f"训练矩阵分解模型时发生错误:\n{str(e)}")

    @error_handler
    def visualize_latent_factors(self):
        """
        可视化潜在因子
        """
        try:
            if not hasattr(self.recommender, 'user_factors') or not hasattr(self.recommender, 'item_factors'):
                return

            # 为简单起见，我们显示前几个潜在因子的分布
            # 使用条形图显示每个因子的平均幅度

            n_factors = min(10, self.recommender.user_factors.shape[1])

            # 计算每个因子的平均绝对值
            user_factor_avgs = np.mean(np.abs(self.recommender.user_factors[:, :n_factors]), axis=0)

            # 准备图表数据
            factor_labels = [f"因子 {i + 1}" for i in range(n_factors)]

            # 设置图表
            self.model_chart.set_title("潜在因子分布")
            self.model_chart.set_labels("潜在因子", "平均权重")
            self.model_chart.set_bar_data(factor_labels, user_factor_avgs)

        except Exception as e:
            logging.error(f"可视化潜在因子时发生错误: {str(e)}")

    @error_handler
    def generate_recommendations(self):
        """
        使用矩阵分解生成推荐
        """
        try:
            # 获取参数
            user_id = int(self.user_combo.currentText())
            n_recommendations = self.recommendations_spin.value()

            # 清空之前的结果
            self.results_table.setRowCount(0)

            # 更新过程文本
            self.process_text.insertHtml(f"""
            <h4>为用户 {user_id} 生成推荐</h4>
            <p>使用矩阵分解模型生成 {n_recommendations} 条推荐</p>
            <hr>
            <p>矩阵分解推荐计算预测评分的公式:</p>
            <div class="formula">
                预测评分(u,i) = P<sub>u</sub> · Q<sub>i</sub><sup>T</sup>
            </div>
            <p>其中:</p>
            <ul>
                <li>P<sub>u</sub>: 用户 u 的潜在因子向量</li>
                <li>Q<sub>i</sub>: 物品 i 的潜在因子向量</li>
            </ul>
            <hr>
            <p>正在生成推荐...</p>
            """)
            QApplication.processEvents()

            # 生成推荐
            self.recommendations = self.recommender.get_matrix_factorization_recommendations(
                user_id=user_id,
                n_recommendations=n_recommendations
            )

            # 显示推荐结果
            self.display_recommendations()

            # 启用保存按钮
            self.save_btn.setEnabled(True)

            # 添加完成信息
            self.process_text.insertHtml("""
            <h4>推荐生成完成</h4>
            <p>已成功生成基于矩阵分解的推荐结果。</p>
            """)

        except Exception as e:
            logging.error(f"生成矩阵分解推荐时发生错误: {str(e)}")
            QMessageBox.critical(self, "错误", f"生成矩阵分解推荐时发生错误:\n{str(e)}")

    @error_handler
    def display_recommendations(self):
        """
        在表格中显示推荐结果
        """
        try:
            if not self.recommendations:
                return

            # 清空表格
            self.results_table.setRowCount(0)

            # 添加推荐结果
            for i, (book_id, book_title, score) in enumerate(self.recommendations):
                self.results_table.insertRow(i)

                # 添加图书ID
                id_item = QTableWidgetItem(str(book_id))
                id_item.setTextAlignment(Qt.AlignCenter)
                self.results_table.setItem(i, 0, id_item)

                # 添加图书标题
                title_item = QTableWidgetItem(book_title)
                self.results_table.setItem(i, 1, title_item)

                # 添加预测评分
                score_item = QTableWidgetItem(f"{score:.2f}")
                score_item.setTextAlignment(Qt.AlignCenter)

                # 根据评分设置背景颜色
                if score >= 4.5:
                    score_item.setBackground(QColor(108, 198, 68, 100))  # 绿色
                elif score >= 4.0:
                    score_item.setBackground(QColor(0, 122, 204, 100))  # 蓝色
                elif score >= 3.5:
                    score_item.setBackground(QColor(225, 151, 76, 100))  # 橙色

                self.results_table.setItem(i, 2, score_item)

            # 调整列宽
            self.results_table.resizeColumnsToContents()

        except Exception as e:
            logging.error(f"显示推荐结果时发生错误: {str(e)}")

    @error_handler
    def save_recommendations(self):
        """
        保存推荐结果到CSV文件
        """
        try:
            if not self.recommendations:
                QMessageBox.warning(self, "警告", "没有推荐结果可保存")
                return

            # 选择保存文件
            file_path, _ = QFileDialog.getSaveFileName(
                self, "保存推荐结果", "", "CSV文件 (*.csv)"
            )

            if not file_path:
                return

            # 保存推荐结果
            self.recommender.save_recommendations(self.recommendations, file_path)

            # 提示成功
            QMessageBox.information(self, "成功", f"推荐结果已保存到:\n{file_path}")

        except Exception as e:
            logging.error(f"保存推荐结果时发生错误: {str(e)}")
            QMessageBox.critical(self, "错误", f"保存推荐结果时发生错误:\n{str(e)}")


# 确保资源目录存在
def ensure_resources_dir():
    """确保资源目录存在"""
    if not os.path.exists('resources'):
        os.makedirs('resources')
    if not os.path.exists('resources/icons'):
        os.makedirs('resources/icons')


# 内嵌资源图标（简单图标生成）
def create_simple_icons():
    """创建简单的图标集"""
    icons = {}

    # 创建一个简单的数据图标
    data_img = QPixmap(64, 64)
    data_img.fill(Qt.transparent)
    painter = QPainter(data_img)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setPen(QPen(QColor("#2D9CDB"), 2))
    painter.setBrush(QBrush(QColor(45, 156, 219, 100)))
    painter.drawRect(10, 15, 44, 35)
    painter.drawLine(10, 25, 54, 25)
    painter.drawLine(20, 25, 20, 50)
    painter.drawLine(30, 25, 30, 50)
    painter.drawLine(40, 25, 40, 50)
    painter.end()
    icons['data'] = QIcon(data_img)

    # 创建一个简单的用户图标
    user_img = QPixmap(64, 64)
    user_img.fill(Qt.transparent)
    painter = QPainter(user_img)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setPen(QPen(QColor("#F2994A"), 2))
    painter.setBrush(QBrush(QColor(242, 153, 74, 100)))
    painter.drawEllipse(22, 15, 20, 20)  # 头部
    painter.drawRect(17, 35, 30, 25)  # 身体
    painter.end()
    icons['user'] = QIcon(user_img)

    # 创建一个简单的书籍图标
    book_img = QPixmap(64, 64)
    book_img.fill(Qt.transparent)
    painter = QPainter(book_img)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setPen(QPen(QColor("#6FCF97"), 2))
    painter.setBrush(QBrush(QColor(111, 207, 151, 100)))
    painter.drawRect(15, 10, 35, 45)
    painter.drawLine(25, 10, 25, 55)
    for i in range(5):
        painter.drawLine(27, 20 + i * 7, 45, 20 + i * 7)
    painter.end()
    icons['book'] = QIcon(book_img)

    # 创建一个简单的评估图标
    eval_img = QPixmap(64, 64)
    eval_img.fill(Qt.transparent)
    painter = QPainter(eval_img)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setPen(QPen(QColor("#BB6BD9"), 2))
    painter.setBrush(QBrush(QColor(187, 107, 217, 100)))
    painter.drawRect(10, 15, 44, 35)
    # 绘制条形图
    painter.drawRect(15, 40, 5, 5)
    painter.drawRect(25, 30, 5, 15)
    painter.drawRect(35, 20, 5, 25)
    painter.drawRect(45, 35, 5, 10)
    painter.end()
    icons['evaluate'] = QIcon(eval_img)

    # 创建一个简单的设置图标
    settings_img = QPixmap(64, 64)
    settings_img.fill(Qt.transparent)
    painter = QPainter(settings_img)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setPen(QPen(QColor("#BDBDBD"), 2))
    painter.setBrush(QBrush(QColor(189, 189, 189, 100)))
    painter.drawEllipse(22, 22, 20, 20)

    # 绘制齿轮
    for i in range(8):
        angle = i * 45
        transform = QTransform()
        transform.translate(32, 32)
        transform.rotate(angle)
        painter.setTransform(transform)
        painter.drawRect(0, -3, 15, 6)
        painter.resetTransform()

    painter.end()
    icons['settings'] = QIcon(settings_img)

    return icons


# VSCode风格主题
class VSCodeTheme:
    """VSCode风格主题类"""

    def __init__(self, is_dark=True):
        """初始化主题

        参数:
            is_dark (bool): 是否使用深色主题
        """
        self.is_dark = is_dark
        self.init_theme()

    def init_theme(self):
        """初始化主题颜色"""
        if self.is_dark:
            # 深色主题
            self.background = QColor(30, 30, 30)
            self.background_light = QColor(37, 37, 38)
            self.background_hover = QColor(44, 44, 44)
            self.sidebar_bg = QColor(51, 51, 51)
            self.text_color = QColor(204, 204, 204)
            self.text_dimmed = QColor(153, 153, 153)
            self.text_highlight = QColor(255, 255, 255)
            self.border_color = QColor(60, 60, 60)
            self.accent_color = QColor(0, 122, 204)
            self.accent_hover = QColor(14, 99, 156)
            self.widget_bg = QColor(45, 45, 45)
            self.selection_bg = QColor(55, 100, 150, 128)
            self.error_color = QColor(224, 108, 117)
            self.success_color = QColor(152, 195, 121)
            self.warning_color = QColor(229, 192, 123)
            self.button_color = QColor(45, 45, 48)
            self.button_hover = QColor(62, 62, 66)
            self.tab_active = QColor(37, 37, 38)
            self.tab_inactive = QColor(45, 45, 45)
            self.focus_border = QColor(0, 120, 215)
        else:
            # 浅色主题
            self.background = QColor(240, 240, 240)
            self.background_light = QColor(250, 250, 250)
            self.background_hover = QColor(230, 230, 230)
            self.sidebar_bg = QColor(225, 225, 225)
            self.text_color = QColor(36, 36, 36)
            self.text_dimmed = QColor(102, 102, 102)
            self.text_highlight = QColor(0, 0, 0)
            self.border_color = QColor(213, 213, 213)
            self.accent_color = QColor(0, 122, 204)
            self.accent_hover = QColor(0, 102, 184)
            self.widget_bg = QColor(255, 255, 255)
            self.selection_bg = QColor(153, 209, 255, 128)
            self.error_color = QColor(205, 49, 49)
            self.success_color = QColor(0, 128, 0)
            self.warning_color = QColor(176, 137, 0)
            self.button_color = QColor(236, 236, 236)
            self.button_hover = QColor(229, 229, 229)
            self.tab_active = QColor(255, 255, 255)
            self.tab_inactive = QColor(245, 245, 245)
            self.focus_border = QColor(0, 120, 215)

    def apply_to_app(self, app):
        """应用主题到整个应用

        参数:
            app (QApplication): 要应用主题的应用实例
        """
        # 设置应用样式表
        app.setStyle(QStyleFactory.create("Fusion"))

        # 创建调色板
        palette = QPalette()
        palette.setColor(QPalette.Window, self.background)
        palette.setColor(QPalette.WindowText, self.text_color)
        palette.setColor(QPalette.Base, self.background_light)
        palette.setColor(QPalette.AlternateBase, self.background)
        palette.setColor(QPalette.ToolTipBase, self.widget_bg)
        palette.setColor(QPalette.ToolTipText, self.text_color)
        palette.setColor(QPalette.Text, self.text_color)
        palette.setColor(QPalette.Button, self.button_color)
        palette.setColor(QPalette.ButtonText, self.text_color)
        palette.setColor(QPalette.BrightText, Qt.red)
        palette.setColor(QPalette.Link, self.accent_color)
        palette.setColor(QPalette.Highlight, self.selection_bg)
        palette.setColor(QPalette.HighlightedText, self.text_highlight)
        palette.setColor(QPalette.Disabled, QPalette.Text, self.text_dimmed)
        palette.setColor(QPalette.Disabled, QPalette.ButtonText, self.text_dimmed)

        app.setPalette(palette)

        # 应用样式表
        stylesheet = f"""
        QMainWindow, QDialog {{
            background-color: {self.background.name()};
            color: {self.text_color.name()};
        }}

        QWidget {{
            background-color: {self.background.name()};
            color: {self.text_color.name()};
        }}

        QTabWidget::pane {{
            border: 1px solid {self.border_color.name()};
            background-color: {self.background_light.name()};
        }}

        QTabBar::tab {{
            background-color: {self.tab_inactive.name()};
            border: 1px solid {self.border_color.name()};
            padding: 6px 12px;
            margin-right: 2px;
        }}

        QTabBar::tab:selected {{
            background-color: {self.tab_active.name()};
            border-bottom-color: {self.tab_active.name()};
        }}

        QTabBar::tab:hover:!selected {{
            background-color: {self.background_hover.name()};
        }}

        QPushButton {{
            background-color: {self.button_color.name()};
            border: 1px solid {self.border_color.name()};
            padding: 5px 15px;
            border-radius: 2px;
        }}

        QPushButton:hover {{
            background-color: {self.button_hover.name()};
        }}

        QPushButton:pressed {{
            background-color: {self.accent_color.name()};
            color: white;
        }}

        QPushButton:disabled {{
            background-color: {self.background.name()};
            color: {self.text_dimmed.name()};
            border: 1px solid {self.border_color.name()};
        }}

        QLineEdit, QTextEdit, QPlainTextEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
            background-color: {self.widget_bg.name()};
            border: 1px solid {self.border_color.name()};
            padding: 3px;
            selection-background-color: {self.selection_bg.name()};
        }}

        QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus, QComboBox:focus {{
            border: 1px solid {self.focus_border.name()};
        }}

        QComboBox::drop-down {{
            border: 0px;
            width: 20px;
        }}

        QComboBox::down-arrow {{
            width: 12px;
            height: 12px;
        }}

        QTableWidget {{
            gridline-color: {self.border_color.name()};
            background-color: {self.background_light.name()};
            outline: 0;
        }}

        QTableWidget::item {{
            padding: 5px;
            border-bottom: 1px solid {self.border_color.name()};
        }}

        QTableWidget::item:selected {{
            background-color: {self.selection_bg.name()};
        }}

        QHeaderView::section {{
            background-color: {self.background.name()};
            padding: 5px;
            border: 1px solid {self.border_color.name()};
            font-weight: bold;
        }}

        QScrollBar:vertical {{
            border: none;
            background: {self.background.name()};
            width: 10px;
            margin: 0px;
        }}

        QScrollBar::handle:vertical {{
            background: {self.button_color.name()};
            min-height: 20px;
            border-radius: 5px;
        }}

        QScrollBar::handle:vertical:hover {{
            background: {self.button_hover.name()};
        }}

        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            height: 0px;
        }}

        QScrollBar:horizontal {{
            border: none;
            background: {self.background.name()};
            height: 10px;
            margin: 0px;
        }}

        QScrollBar::handle:horizontal {{
            background: {self.button_color.name()};
            min-width: 20px;
            border-radius: 5px;
        }}

        QScrollBar::handle:horizontal:hover {{
            background: {self.button_hover.name()};
        }}

        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
            width: 0px;
        }}

        QGroupBox {{
            border: 1px solid {self.border_color.name()};
            margin-top: 1ex;
            background-color: {self.background_light.name()};
            border-radius: 3px;
        }}

        QGroupBox::title {{
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 5px;
            color: {self.text_color.name()};
        }}

        QStatusBar {{
            background-color: {self.background.name()};
            color: {self.text_color.name()};
            border-top: 1px solid {self.border_color.name()};
        }}

        QSplitter::handle {{
            background-color: {self.border_color.name()};
        }}

        QToolButton {{
            background-color: transparent;
            border: none;
            padding: 3px;
            border-radius: 2px;
        }}

        QToolButton:hover {{
            background-color: {self.button_hover.name()};
        }}

        QListWidget {{
            background-color: {self.sidebar_bg.name()};
            border: none;
            outline: none;
        }}

        QListWidget::item {{
            padding: 8px 5px;
            border-left: 3px solid transparent;
        }}

        QListWidget::item:selected {{
            background-color: {self.background_hover.name()};
            border-left: 3px solid {self.accent_color.name()};
        }}

        QListWidget::item:hover:!selected {{
            background-color: {self.background_hover.name()};
        }}

        QProgressBar {{
            border: 1px solid {self.border_color.name()};
            border-radius: 2px;
            text-align: center;
            background-color: {self.background_light.name()};
        }}

        QProgressBar::chunk {{
            background-color: {self.accent_color.name()};
            width: 10px;
        }}

        QMenuBar {{
            background-color: {self.background.name()};
            border-bottom: 1px solid {self.border_color.name()};
        }}

        QMenuBar::item {{
            spacing: 5px;
            padding: 3px 10px;
            background: transparent;
        }}

        QMenuBar::item:selected {{
            background-color: {self.selection_bg.name()};
        }}

        QMenu {{
            background-color: {self.background_light.name()};
            border: 1px solid {self.border_color.name()};
        }}

        QMenu::item {{
            padding: 5px 20px 5px 20px;
        }}

        QMenu::item:selected {{
            background-color: {self.selection_bg.name()};
        }}

        QCheckBox {{
            spacing: 5px;
        }}

        QRadioButton {{
            spacing: 5px;
        }}
        """

        app.setStyleSheet(stylesheet)


# 自定义的VSCode风格按钮
class VSCodeButton(QPushButton):
    """VSCode风格的按钮"""

    def __init__(self, text="", icon=None, parent=None):
        """初始化按钮

        参数:
            text (str): 按钮文本
            icon (QIcon): 按钮图标
            parent (QWidget): 父窗口
        """
        super(VSCodeButton, self).__init__(text, parent)
        if icon:
            self.setIcon(icon)
        self.setMinimumHeight(28)

        # 默认样式
        self.setCursor(QCursor(Qt.PointingHandCursor))


# 卡片组件
class CardWidget(QFrame):
    """卡片样式组件，用于包装内容"""

    def __init__(self, title="", parent=None):
        """初始化卡片组件

        参数:
            title (str): 卡片标题
            parent (QWidget): 父窗口
        """
        super(CardWidget, self).__init__(parent)

        # 设置基本样式
        self.setFrameShape(QFrame.StyledPanel)
        self.setFrameShadow(QFrame.Raised)
        self.setObjectName("card")

        # 创建布局
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        self.main_layout.setSpacing(10)

        # 标题栏
        if title:
            self.title_label = QLabel(title)
            font = self.title_label.font()
            font.setBold(True)
            font.setPointSize(11)
            self.title_label.setFont(font)
            self.main_layout.addWidget(self.title_label)

            # 添加分隔线
            self.separator = QFrame()
            self.separator.setFrameShape(QFrame.HLine)
            self.separator.setFrameShadow(QFrame.Sunken)
            self.main_layout.addWidget(self.separator)

        # 内容布局
        self.content_layout = QVBoxLayout()
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(8)
        self.main_layout.addLayout(self.content_layout)

        # 添加阴影效果
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(15)
        shadow.setColor(QColor(0, 0, 0, 50))
        shadow.setOffset(0, 2)
        self.setGraphicsEffect(shadow)

    def add_widget(self, widget):
        """添加控件到卡片内容区

        参数:
            widget (QWidget): 要添加的控件
        """
        self.content_layout.addWidget(widget)

    def add_layout(self, layout):
        """添加布局到卡片内容区

        参数:
            layout (QLayout): 要添加的布局
        """
        self.content_layout.addLayout(layout)


# 侧边栏选项
class SidebarItem(QFrame):
    """侧边栏选项组件"""

    clicked = Signal(str)  # 点击信号

    def __init__(self, icon, text, item_id, parent=None):
        """初始化侧边栏选项

        参数:
            icon (QIcon): 选项图标
            text (str): 选项文本
            item_id (str): 选项ID
            parent (QWidget): 父窗口
        """
        super(SidebarItem, self).__init__(parent)
        self.item_id = item_id
        self.setObjectName("sidebarItem")
        self.setCursor(QCursor(Qt.PointingHandCursor))

        # 布局
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(10)

        # 图标
        self.icon_label = QLabel()
        icon_pixmap = icon.pixmap(QSize(20, 20))
        self.icon_label.setPixmap(icon_pixmap)
        layout.addWidget(self.icon_label)

        # 文本
        self.text_label = QLabel(text)
        layout.addWidget(self.text_label)

        # 占位符
        layout.addStretch()

    def mousePressEvent(self, event):
        """鼠标按下事件处理"""
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.item_id)
        super(SidebarItem, self).mousePressEvent(event)


# 侧边栏
class Sidebar(QFrame):
    """VSCode风格侧边栏"""

    item_selected = Signal(str)  # 选项选择信号

    def __init__(self, parent=None):
        """初始化侧边栏

        参数:
            parent (QWidget): 父窗口
        """
        super(Sidebar, self).__init__(parent)
        self.setObjectName("sidebar")
        self.setMinimumWidth(200)
        self.setMaximumWidth(300)

        # 创建布局
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 10, 0, 10)
        self.layout.setSpacing(5)

        # 标题
        self.title_label = QLabel("图书馆推荐系统")
        font = self.title_label.font()
        font.setBold(True)
        font.setPointSize(12)
        self.title_label.setFont(font)
        self.title_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.title_label)

        # 图标
        self.application_logo = QLabel()
        self.application_logo.setAlignment(Qt.AlignCenter)
        # 这里可以加载应用LOGO
        self.layout.addWidget(self.application_logo)

        # 分隔线
        self.separator = QFrame()
        self.separator.setFrameShape(QFrame.HLine)
        self.separator.setFrameShadow(QFrame.Sunken)
        self.layout.addWidget(self.separator)

        # 侧边栏项目容器
        self.items_container = QWidget()
        self.items_layout = QVBoxLayout(self.items_container)
        self.items_layout.setContentsMargins(0, 5, 0, 5)
        self.items_layout.setSpacing(0)
        self.layout.addWidget(self.items_container)

        # 添加伸缩器
        self.layout.addStretch()

        # 当前选中的ID
        self.selected_id = None

        # 项目映射表
        self.items = {}

    def add_item(self, icon, text, item_id):
        """添加侧边栏选项

        参数:
            icon (QIcon): 选项图标
            text (str): 选项文本
            item_id (str): 选项ID
        """
        item = SidebarItem(icon, text, item_id)
        self.items_layout.addWidget(item)
        self.items[item_id] = item

        # 连接信号
        item.clicked.connect(self._on_item_clicked)

    def select_item(self, item_id):
        """选中指定的选项

        参数:
            item_id (str): 要选中的选项ID
        """
        if item_id in self.items:
            self._on_item_clicked(item_id)

    def _on_item_clicked(self, item_id):
        """处理项目点击事件

        参数:
            item_id (str): 被点击的选项ID
        """
        # 更新样式
        if self.selected_id:
            self.items[self.selected_id].setStyleSheet("")

        self.selected_id = item_id
        self.items[item_id].setStyleSheet("""
            background-color: rgba(255, 255, 255, 0.1);
            border-left: 3px solid #007ACC;
        """)

        # 发射信号
        self.item_selected.emit(item_id)


# 改进的图表组件
class ModernChartWidget(QFrame):
    """现代风格图表组件，提供基本的数据可视化功能"""

    def __init__(self, parent=None):
        """
        初始化图表组件

        参数:
            parent: 父窗口
        """
        super(ModernChartWidget, self).__init__(parent)
        self.setMinimumSize(300, 200)
        self.setFrameShape(QFrame.StyledPanel)
        self.setFrameShadow(QFrame.Sunken)

        # 图表数据
        self.title = ""
        self.x_label = ""
        self.y_label = ""
        self.data = None  # 将存储(x, y)数据点或类似结构
        self.chart_type = "bar"  # 可选: "bar", "line", "heatmap"
        self.color_map = [
            QColor(0, 122, 204),  # VS Code蓝
            QColor(108, 198, 68),  # 绿色
            QColor(225, 151, 76),  # 橙色
            QColor(186, 107, 217),  # 紫色
            QColor(241, 143, 1)  # 黄色
        ]
        self.background_color = QColor(30, 30, 30)
        self.grid_color = QColor(60, 60, 60)
        self.text_color = QColor(204, 204, 204)

    def set_title(self, title):
        """设置图表标题"""
        self.title = title
        self.update()

    def set_labels(self, x_label, y_label):
        """设置坐标轴标签"""
        self.x_label = x_label
        self.y_label = y_label
        self.update()

    def set_bar_data(self, categories, values):
        """
        设置条形图数据

        参数:
            categories: 类别标签列表
            values: 对应的数值列表
        """
        self.chart_type = "bar"
        self.data = list(zip(categories, values))
        self.update()

    def set_heatmap_data(self, matrix, row_labels=None, col_labels=None):
        """
        设置热力图数据

        参数:
            matrix: 2D numpy数组或列表的列表
            row_labels: 行标签列表
            col_labels: 列标签列表
        """
        self.chart_type = "heatmap"
        self.data = {
            'matrix': matrix,
            'row_labels': row_labels if row_labels is not None else [],
            'col_labels': col_labels if col_labels is not None else []
        }
        self.update()

    def paintEvent(self, event):
        """处理绘图事件"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # 绘制背景
        painter.fillRect(event.rect(), self.background_color)

        # 设置画笔和字体
        painter.setPen(QPen(self.text_color, 1))
        font = painter.font()
        font.setPointSize(10)
        painter.setFont(font)

        # 获取绘图区域
        width = self.width()
        height = self.height()
        margin = 40  # 边距

        # 绘制标题
        if self.title:
            font.setPointSize(12)
            font.setBold(True)
            painter.setFont(font)
            painter.drawText(QRectF(0, 5, width, 25), Qt.AlignCenter, self.title)
            font.setPointSize(10)
            font.setBold(False)
            painter.setFont(font)

        # 图表绘制区域
        chart_rect = QRectF(margin, margin, width - 2 * margin, height - 2 * margin)

        # 绘制网格
        painter.setPen(QPen(self.grid_color, 0.5, Qt.DotLine))
        for i in range(5):
            y = chart_rect.top() + i * chart_rect.height() / 4
            painter.drawLine(QPointF(chart_rect.left(), y), QPointF(chart_rect.right(), y))

        painter.setPen(QPen(self.text_color, 1))

        # 根据图表类型调用相应的绘制方法
        if self.chart_type == "bar" and self.data:
            self._draw_bar_chart(painter, chart_rect)
        elif self.chart_type == "heatmap" and self.data:
            self._draw_heatmap(painter, chart_rect)

        # 绘制坐标轴
        painter.setPen(QPen(self.text_color, 1.5))
        painter.drawLine(
            QPointF(chart_rect.left(), chart_rect.top()),
            QPointF(chart_rect.left(), chart_rect.bottom())
        )
        painter.drawLine(
            QPointF(chart_rect.left(), chart_rect.bottom()),
            QPointF(chart_rect.right(), chart_rect.bottom())
        )

        # 绘制坐标轴标签
        if self.x_label:
            painter.drawText(
                QRectF(margin, height - 25, width - 2 * margin, 20),
                Qt.AlignCenter,
                self.x_label
            )
        if self.y_label:
            # 保存当前状态
            painter.save()
            # 旋转绘图坐标系以绘制垂直文本
            painter.translate(15, height / 2)
            painter.rotate(-90)
            painter.drawText(
                QRectF(-50, 0, 100, 20),
                Qt.AlignCenter,
                self.y_label
            )
            # 恢复状态
            painter.restore()

    def _draw_bar_chart(self, painter, rect):
        """绘制条形图"""
        if not self.data:
            return

        # 获取数据
        categories, values = zip(*self.data)
        num_bars = len(values)

        # 计算条形图设置
        bar_width = rect.width() / (num_bars * 2)
        max_value = max(values) if values else 0
        scale_factor = rect.height() / max_value if max_value > 0 else 0

        # 绘制Y轴刻度
        num_ticks = 5
        for i in range(num_ticks + 1):
            y_value = max_value * i / num_ticks
            y_pos = rect.bottom() - y_value * scale_factor

            # 绘制刻度线
            painter.drawLine(
                QPointF(rect.left() - 5, y_pos),
                QPointF(rect.left(), y_pos)
            )

            # 绘制刻度值
            if max_value < 1:
                # 小数显示更多位数
                tick_text = f"{y_value:.2f}"
            else:
                tick_text = f"{int(y_value) if y_value.is_integer() else y_value:.1f}"

            painter.drawText(
                QRectF(rect.left() - 35, y_pos - 10, 30, 20),
                Qt.AlignRight | Qt.AlignVCenter,
                tick_text
            )

        # 绘制条形
        for i, (category, value) in enumerate(self.data):
            # 计算条形位置
            x = rect.left() + (i + 0.5) * rect.width() / num_bars
            bar_height = value * scale_factor

            # 创建渐变填充
            gradient = QLinearGradient(
                x - bar_width / 2, rect.bottom() - bar_height,
                x - bar_width / 2, rect.bottom()
            )
            color = self.color_map[i % len(self.color_map)]
            gradient.setColorAt(0, color.lighter(130))
            gradient.setColorAt(1, color)

            # 绘制条形
            bar_rect = QRectF(
                x - bar_width / 2,
                rect.bottom() - bar_height,
                bar_width,
                bar_height
            )

            # 添加圆角
            path = QPainterPath()
            path.addRoundedRect(bar_rect, 3, 3)

            painter.fillPath(path, gradient)
            painter.setPen(QPen(color.darker(110), 1))
            painter.drawPath(path)

            # 绘制数值
            if isinstance(value, float):
                value_text = f"{int(value) if value.is_integer() else value:.1f}"
            else:
                value_text = str(value)

            painter.setPen(QPen(self.text_color, 1))
            painter.drawText(
                QRectF(x - 20, rect.bottom() - bar_height - 20, 40, 20),
                Qt.AlignCenter,
                value_text
            )

            # 绘制类别标签
            painter.drawText(
                QRectF(x - 30, rect.bottom() + 5, 60, 20),
                Qt.AlignCenter,
                str(category)
            )

    def _draw_heatmap(self, painter, rect):
        """绘制热力图"""
        if not self.data or 'matrix' not in self.data:
            return

        matrix = self.data['matrix']
        row_labels = self.data['row_labels']
        col_labels = self.data['col_labels']

        # 处理numpy数组
        if hasattr(matrix, 'tolist'):
            matrix = matrix.tolist()

        # 获取矩阵尺寸
        rows = len(matrix)
        cols = len(matrix[0]) if rows > 0 else 0

        if rows == 0 or cols == 0:
            return

        # 计算单元格大小
        cell_width = rect.width() / cols
        cell_height = rect.height() / rows

        # 获取矩阵值的范围
        flat_values = [val for row in matrix for val in row]
        min_val = min(flat_values)
        max_val = max(flat_values)
        val_range = max_val - min_val if max_val > min_val else 1

        # 绘制热力图
        for i in range(rows):
            for j in range(cols):
                # 计算归一化的值 (0-1范围)
                normalized_value = (matrix[i][j] - min_val) / val_range

                # 生成颜色
                if normalized_value < 0.5:
                    # 蓝色到白色
                    r = int(255 * (2 * normalized_value))
                    g = int(255 * (2 * normalized_value))
                    b = 255
                else:
                    # 白色到蓝色 (VS Code风格)
                    r = int(255 * (2 - 2 * normalized_value))
                    g = int(255 * (2 - 2 * normalized_value))
                    b = 255

                color = QColor(r, g, b)

                # 绘制单元格
                cell_rect = QRectF(
                    rect.left() + j * cell_width,
                    rect.top() + i * cell_height,
                    cell_width,
                    cell_height
                )

                # 圆角矩形
                path = QPainterPath()
                path.addRoundedRect(cell_rect, 2, 2)

                painter.fillPath(path, QBrush(color))
                painter.setPen(QPen(Qt.black, 0.5))
                painter.drawPath(path)

                # 如果单元格足够大，绘制数值
                if cell_width > 30 and cell_height > 15:
                    painter.setPen(QPen(Qt.black if normalized_value > 0.5 else Qt.white, 1))
                    value_text = f"{matrix[i][j]:.2f}"
                    painter.drawText(
                        cell_rect,
                        Qt.AlignCenter,
                        value_text
                    )

        # 绘制行标签
        if row_labels and len(row_labels) == rows:
            for i, label in enumerate(row_labels):
                label_rect = QRectF(
                    rect.left() - 60,
                    rect.top() + i * cell_height,
                    55,
                    cell_height
                )
                painter.setPen(QPen(self.text_color, 1))
                painter.drawText(
                    label_rect,
                    Qt.AlignRight | Qt.AlignVCenter,
                    str(label)
                )

        # 绘制列标签
        if col_labels and len(col_labels) == cols:
            for j, label in enumerate(col_labels):
                label_rect = QRectF(
                    rect.left() + j * cell_width,
                    rect.top() - 20,
                    cell_width,
                    15
                )
                painter.setPen(QPen(self.text_color, 1))
                painter.drawText(
                    label_rect,
                    Qt.AlignCenter,
                    str(label)
                )


# 改进的数据加载窗口
class LoadingDataWidget(QWidget):
    """
    数据加载与摘要展示窗口
    """
    # 定义信号
    data_loaded_signal = Signal(bool)  # 数据加载完成信号

    def __init__(self, recommender, theme, icons, parent=None):
        """
        初始化窗口

        参数:
            recommender: LibraryRecommender实例
            theme: 主题对象
            icons: 图标字典
            parent: 父窗口
        """
        super(LoadingDataWidget, self).__init__(parent)
        self.recommender = recommender
        self.theme = theme
        self.icons = icons
        self.init_ui()

    def init_ui(self):
        """
        初始化UI组件
        """
        # 创建主布局
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(20)

        # 顶部操作栏
        top_bar = QHBoxLayout()

        # 标题
        title_label = QLabel("数据管理")
        font = title_label.font()
        font.setPointSize(16)
        font.setBold(True)
        title_label.setFont(font)
        top_bar.addWidget(title_label)

        top_bar.addStretch()

        # 快捷操作按钮
        self.create_demo_btn = VSCodeButton("创建示例数据", self.icons['data'])
        self.create_demo_btn.clicked.connect(self.create_demo_data)
        top_bar.addWidget(self.create_demo_btn)

        self.load_btn = VSCodeButton("加载数据", self.icons['data'])
        self.load_btn.clicked.connect(self.load_data)
        top_bar.addWidget(self.load_btn)

        main_layout.addLayout(top_bar)

        # 文件加载卡片
        load_card = CardWidget("数据文件配置")
        load_layout = QGridLayout()
        load_layout.setColumnStretch(1, 1)  # 第二列（文件路径输入框）可以伸展

        # 用户数据
        self.user_path_label = QLabel("用户数据:")
        self.user_path_edit = QLineEdit()
        self.user_path_edit.setPlaceholderText("选择用户数据文件...")
        self.user_browse_btn = QPushButton("浏览...")
        self.user_browse_btn.clicked.connect(lambda: self.browse_file("用户数据 (*.csv)", self.user_path_edit))

        # 图书数据
        self.book_path_label = QLabel("图书数据:")
        self.book_path_edit = QLineEdit()
        self.book_path_edit.setPlaceholderText("选择图书数据文件...")
        self.book_browse_btn = QPushButton("浏览...")
        self.book_browse_btn.clicked.connect(lambda: self.browse_file("图书数据 (*.csv)", self.book_path_edit))

        # 评分数据
        self.rating_path_label = QLabel("评分数据:")
        self.rating_path_edit = QLineEdit()
        self.rating_path_edit.setPlaceholderText("选择评分数据文件...")
        self.rating_browse_btn = QPushButton("浏览...")
        self.rating_browse_btn.clicked.connect(lambda: self.browse_file("评分数据 (*.csv)", self.rating_path_edit))

        # 添加到布局
        load_layout.addWidget(self.user_path_label, 0, 0)
        load_layout.addWidget(self.user_path_edit, 0, 1)
        load_layout.addWidget(self.user_browse_btn, 0, 2)

        load_layout.addWidget(self.book_path_label, 1, 0)
        load_layout.addWidget(self.book_path_edit, 1, 1)
        load_layout.addWidget(self.book_browse_btn, 1, 2)

        load_layout.addWidget(self.rating_path_label, 2, 0)
        load_layout.addWidget(self.rating_path_edit, 2, 1)
        load_layout.addWidget(self.rating_browse_btn, 2, 2)

        load_card.add_layout(load_layout)
        main_layout.addWidget(load_card)

        # 数据摘要和可视化区域
        data_view_splitter = QSplitter(Qt.Horizontal)

        # 左侧摘要文本区
        summary_card = CardWidget("数据摘要统计")
        summary_layout = QVBoxLayout()

        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)

        summary_layout.addWidget(self.summary_text)
        summary_card.add_layout(summary_layout)
        data_view_splitter.addWidget(summary_card)

        # 右侧可视化区域
        chart_splitter = QSplitter(Qt.Vertical)

        # 用户评分分布可视化
        user_ratings_card = CardWidget("用户评分数量分布")
        user_ratings_layout = QVBoxLayout()
        self.user_ratings_chart = ModernChartWidget()
        user_ratings_layout.addWidget(self.user_ratings_chart)
        user_ratings_card.add_layout(user_ratings_layout)
        chart_splitter.addWidget(user_ratings_card)

        # 评分分布可视化
        rating_dist_card = CardWidget("评分分布")
        rating_dist_layout = QVBoxLayout()
        self.rating_dist_chart = ModernChartWidget()
        rating_dist_layout.addWidget(self.rating_dist_chart)
        rating_dist_card.add_layout(rating_dist_layout)
        chart_splitter.addWidget(rating_dist_card)

        data_view_splitter.addWidget(chart_splitter)

        # 调整分割器比例
        data_view_splitter.setSizes([400, 600])
        chart_splitter.setSizes([300, 300])

        main_layout.addWidget(data_view_splitter, 1)  # 1是伸展因子，让这个区域占用更多空间

    @error_handler
    def browse_file(self, filter_text, line_edit):
        """
        打开文件浏览对话框

        参数:
            filter_text: 文件过滤器文本
            line_edit: 用于显示所选文件路径的QLineEdit
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择文件", "", filter_text
        )
        if file_path:
            line_edit.setText(file_path)

    @error_handler
    def create_demo_data(self):
        """
        创建演示数据
        """
        try:
            # 显示进度对话框
            progress_dialog = QDialog(self)
            progress_dialog.setWindowTitle("正在创建")
            progress_dialog.setFixedSize(300, 100)

            dialog_layout = QVBoxLayout(progress_dialog)

            info_label = QLabel("正在创建示例数据...", progress_dialog)
            dialog_layout.addWidget(info_label)

            progress_bar = QProgressBar(progress_dialog)
            progress_bar.setRange(0, 0)  # 设置为不确定模式
            dialog_layout.addWidget(progress_bar)

            progress_dialog.show()
            QApplication.processEvents()

            # 创建示例数据
            create_demo_data()

            # 关闭进度对话框
            progress_dialog.close()

            # 更新文件路径
            demo_dir = os.path.abspath('demo_data')
            self.user_path_edit.setText(os.path.join(demo_dir, 'users.csv'))
            self.book_path_edit.setText(os.path.join(demo_dir, 'books.csv'))
            self.rating_path_edit.setText(os.path.join(demo_dir, 'ratings.csv'))

            # 提示成功
            QMessageBox.information(self, "成功", "示例数据创建成功！数据文件已保存到demo_data目录。")

        except Exception as e:
            logging.error(f"创建示例数据时发生错误: {str(e)}")
            QMessageBox.critical(self, "错误", f"创建示例数据时发生错误:\n{str(e)}")

    @error_handler
    def load_data(self):
        """
        加载用户选择的数据文件
        """
        try:
            # 获取文件路径
            user_path = self.user_path_edit.text()
            book_path = self.book_path_edit.text()
            rating_path = self.rating_path_edit.text()

            # 检查文件是否存在
            for path, desc in [(user_path, "用户数据"),
                               (book_path, "图书数据"),
                               (rating_path, "评分数据")]:
                if not os.path.exists(path):
                    QMessageBox.warning(self, "警告", f"{desc}文件不存在: {path}")
                    return

            # 显示进度对话框
            progress_dialog = QDialog(self)
            progress_dialog.setWindowTitle("正在加载")
            progress_dialog.setFixedSize(300, 100)

            dialog_layout = QVBoxLayout(progress_dialog)

            info_label = QLabel("正在加载数据...", progress_dialog)
            dialog_layout.addWidget(info_label)

            progress_bar = QProgressBar(progress_dialog)
            progress_bar.setRange(0, 0)  # 设置为不确定模式
            dialog_layout.addWidget(progress_bar)

            progress_dialog.show()
            QApplication.processEvents()

            # 加载数据
            self.recommender.load_data(user_path, book_path, rating_path)

            # 关闭进度对话框
            progress_dialog.close()

            # 显示数据摘要
            self.update_summary()

            # 发送数据加载完成信号
            self.data_loaded_signal.emit(True)

            # 提示成功
            QMessageBox.information(self, "成功", "数据加载成功！")

        except Exception as e:
            logging.error(f"加载数据时发生错误: {str(e)}")
            QMessageBox.critical(self, "错误", f"加载数据时发生错误:\n{str(e)}")
            self.data_loaded_signal.emit(False)

    @error_handler
    def update_summary(self):
        """
        更新数据摘要信息和可视化
        """
        try:
            # 获取数据摘要
            summary = self.recommender.data_summary()

            # 用单行CSS字符串避免换行和缩进问题
            css = "<style>table{border-collapse:collapse;width:100%;}th,td{padding:8px;text-align:left;border-bottom:1px solid #ddd;}th{background-color:rgba(0,122,204,0.2);}h3{color:#0078D4;margin-top:16px;margin-bottom:8px;}</style>"

            # 构造HTML内容
            html_content = f"""
{css}
<h3>基本统计</h3>
<table>
<tr><th>统计项</th><th>数值</th></tr>
<tr><td>用户数量</td><td>{summary.get('users_count', 'N/A')}</td></tr>
<tr><td>图书数量</td><td>{summary.get('books_count', 'N/A')}</td></tr>
<tr><td>评分数量</td><td>{summary.get('ratings_count', 'N/A')}</td></tr>
<tr><td>平均评分</td><td>{summary.get('average_rating', 0):.2f}</td></tr>
<tr><td>数据稀疏度</td><td>{summary.get('sparsity', 0):.4f} (值越大表示数据越稀疏)</td></tr>
</table>

<h3>每用户评分统计</h3>
<table>
<tr><th>统计项</th><th>数值</th></tr>
<tr><td>最少评分</td><td>{summary.get('ratings_per_user', {}).get('min', 'N/A')}</td></tr>
<tr><td>最多评分</td><td>{summary.get('ratings_per_user', {}).get('max', 'N/A')}</td></tr>
<tr><td>平均评分数</td><td>{summary.get('ratings_per_user', {}).get('mean', 0):.2f}</td></tr>
</table>

<h3>每本书评分统计</h3>
<table>
<tr><th>统计项</th><th>数值</th></tr>
<tr><td>最少评分</td><td>{summary.get('ratings_per_book', {}).get('min', 'N/A')}</td></tr>
<tr><td>最多评分</td><td>{summary.get('ratings_per_book', {}).get('max', 'N/A')}</td></tr>
<tr><td>平均评分数</td><td>{summary.get('ratings_per_book', {}).get('mean', 0):.2f}</td></tr>
</table>
"""
            # 设置HTML内容 - 试用纯文本而不是HTML
            self.summary_text.setText(f"""
基本统计:
* 用户数量: {summary.get('users_count', 'N/A')}
* 图书数量: {summary.get('books_count', 'N/A')}
* 评分数量: {summary.get('ratings_count', 'N/A')}
* 平均评分: {summary.get('average_rating', 0):.2f}
* 数据稀疏度: {summary.get('sparsity', 0):.4f} (值越大表示数据越稀疏)

每用户评分统计:
* 最少评分: {summary.get('ratings_per_user', {}).get('min', 'N/A')}
* 最多评分: {summary.get('ratings_per_user', {}).get('max', 'N/A')}
* 平均评分数: {summary.get('ratings_per_user', {}).get('mean', 0):.2f}

每本书评分统计:
* 最少评分: {summary.get('ratings_per_book', {}).get('min', 'N/A')}
* 最多评分: {summary.get('ratings_per_book', {}).get('max', 'N/A')}
* 平均评分数: {summary.get('ratings_per_book', {}).get('mean', 0):.2f}
""")

            self.summary_text.setHtml(html_content)

            # 绘制用户评分数量分布
            self.plot_user_ratings_dist()

            # 绘制评分分布
            self.plot_rating_dist(summary.get('rating_distribution', {}))

        except Exception as e:
            logging.error(f"更新数据摘要时发生错误: {str(e)}")
            QMessageBox.critical(self, "错误", f"更新数据摘要时发生错误:\n{str(e)}")

    @error_handler
    def plot_user_ratings_dist(self):
        """
        绘制用户评分数量分布图
        """
        try:
            if self.recommender.ratings_df is None:
                return

            # 获取每个用户的评分数量
            user_rating_counts = self.recommender.ratings_df['user_id'].value_counts()

            # 创建直方图数据
            # 简化：将用户分成评分数量区间
            bins = [0, 5, 10, 15, 20, 25, 30, 35, 40]
            bin_counts = [0] * (len(bins) - 1)

            for count in user_rating_counts:
                for i in range(len(bins) - 1):
                    if bins[i] <= count < bins[i + 1]:
                        bin_counts[i] += 1
                        break

            # 构建标签
            labels = [f"{bins[i]}-{bins[i + 1]}" for i in range(len(bins) - 1)]

            # 绘制图表
            self.user_ratings_chart.set_title("用户评分数量分布")
            self.user_ratings_chart.set_labels("评分数量范围", "用户数")
            self.user_ratings_chart.set_bar_data(labels, bin_counts)

        except Exception as e:
            logging.error(f"绘制用户评分分布图时发生错误: {str(e)}")

    @error_handler
    def plot_rating_dist(self, rating_dist):
        """
        绘制评分分布图

        参数:
            rating_dist: 评分分布字典 {评分: 数量}
        """
        try:
            # 准备数据
            ratings = sorted(rating_dist.keys())
            counts = [rating_dist[r] for r in ratings]

            # 绘制条形图
            self.rating_dist_chart.set_title("评分分布")
            self.rating_dist_chart.set_labels("评分", "数量")
            self.rating_dist_chart.set_bar_data(ratings, counts)

        except Exception as e:
            logging.error(f"绘制评分分布图时发生错误: {str(e)}")


# 改进的用户推荐界面
class UserBasedRecommendationWidget(QWidget):
    """
    基于用户的推荐窗口
    """

    def __init__(self, recommender, theme, icons, parent=None):
        """
        初始化窗口

        参数:
            recommender: LibraryRecommender实例
            theme: 主题对象
            icons: 图标字典
            parent: 父窗口
        """
        super(UserBasedRecommendationWidget, self).__init__(parent)
        self.recommender = recommender
        self.theme = theme
        self.icons = icons
        self.similarity_matrix = None
        self.recommendations = None
        self.init_ui()

    def init_ui(self):
        """
        初始化UI组件
        """
        # 创建主布局
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(20)

        # 顶部操作栏
        top_bar = QHBoxLayout()

        # 标题
        title_label = QLabel("基于用户的协同过滤推荐")
        font = title_label.font()
        font.setPointSize(16)
        font.setBold(True)
        title_label.setFont(font)
        top_bar.addWidget(title_label)

        top_bar.addStretch()

        # 操作按钮
        self.calc_similarity_btn = VSCodeButton("计算用户相似度", self.icons['user'])
        self.calc_similarity_btn.clicked.connect(self.calculate_similarity)
        top_bar.addWidget(self.calc_similarity_btn)

        main_layout.addLayout(top_bar)

        # 上半部分：控制面板和相似度矩阵可视化
        top_section = QSplitter(Qt.Horizontal)

        # 控制面板
        control_card = CardWidget("推荐参数设置")
        control_layout = QVBoxLayout()

        # 参数设置
        params_group = QWidget()
        params_layout = QGridLayout(params_group)
        params_layout.setColumnStretch(1, 1)  # 设置第二列可伸展

        # 用户选择
        self.user_label = QLabel("选择用户:")
        self.user_combo = QComboBox()

        # 参数设置
        self.neighbors_label = QLabel("邻居数量:")
        self.neighbors_spin = QSpinBox()
        self.neighbors_spin.setRange(1, 20)
        self.neighbors_spin.setValue(5)
        self.neighbors_spin.setToolTip("考虑的相似用户数量")

        self.recommendations_label = QLabel("推荐数量:")
        self.recommendations_spin = QSpinBox()
        self.recommendations_spin.setRange(1, 20)
        self.recommendations_spin.setValue(10)
        self.recommendations_spin.setToolTip("要生成的推荐图书数量")

        # 添加到布局
        params_layout.addWidget(self.user_label, 0, 0)
        params_layout.addWidget(self.user_combo, 0, 1)

        params_layout.addWidget(self.neighbors_label, 1, 0)
        params_layout.addWidget(self.neighbors_spin, 1, 1)

        params_layout.addWidget(self.recommendations_label, 2, 0)
        params_layout.addWidget(self.recommendations_spin, 2, 1)

        control_layout.addWidget(params_group)

        # 添加分隔线
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        control_layout.addWidget(separator)

        # 操作按钮区
        buttons_layout = QVBoxLayout()

        self.recommend_btn = VSCodeButton("生成推荐", self.icons['book'])
        self.recommend_btn.clicked.connect(self.generate_recommendations)
        self.recommend_btn.setEnabled(False)
        buttons_layout.addWidget(self.recommend_btn)

        self.save_btn = VSCodeButton("保存推荐结果", None)
        self.save_btn.clicked.connect(self.save_recommendations)
        self.save_btn.setEnabled(False)
        buttons_layout.addWidget(self.save_btn)

        control_layout.addLayout(buttons_layout)

        # 添加伸缩器
        control_layout.addStretch()

        control_card.add_layout(control_layout)
        top_section.addWidget(control_card)

        # 相似度矩阵可视化
        similarity_card = CardWidget("用户相似度矩阵")
        similarity_layout = QVBoxLayout()

        self.similarity_chart = ModernChartWidget()
        similarity_layout.addWidget(self.similarity_chart)

        similarity_card.add_layout(similarity_layout)
        top_section.addWidget(similarity_card)

        # 设置分割比例
        top_section.setSizes([300, 700])

        main_layout.addWidget(top_section, 1)  # 1是伸展因子

        # 下半部分：推荐过程可视化和结果
        bottom_section = QSplitter(Qt.Horizontal)

        # 推荐过程可视化
        process_card = CardWidget("推荐过程")
        process_layout = QVBoxLayout()

        self.process_text = QTextEdit()
        self.process_text.setReadOnly(True)

        process_layout.addWidget(self.process_text)

        process_card.add_layout(process_layout)
        bottom_section.addWidget(process_card)

        # 推荐结果
        results_card = CardWidget("推荐结果")
        results_layout = QVBoxLayout()

        self.results_table = QTableWidget()
        self.results_table.setColumnCount(3)
        self.results_table.setHorizontalHeaderLabels(["图书ID", "图书标题", "预测评分"])
        self.results_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)

        results_layout.addWidget(self.results_table)

        results_card.add_layout(results_layout)
        bottom_section.addWidget(results_card)

        # 设置分割比例
        bottom_section.setSizes([500, 500])

        main_layout.addWidget(bottom_section, 1)  # 1是伸展因子

    def update_user_list(self):
        """
        更新用户下拉列表
        """
        try:
            self.user_combo.clear()

            if self.recommender.users_df is not None:
                for user_id in sorted(self.recommender.users_df['user_id'].unique()):
                    self.user_combo.addItem(str(user_id))
        except Exception as e:
            logging.error(f"更新用户列表时发生错误: {str(e)}")
            QMessageBox.critical(self, "错误", f"更新用户列表时发生错误:\n{str(e)}")

    @error_handler
    def calculate_similarity(self):
        """
        计算用户相似度并可视化
        """
        try:
            # 显示进度对话框
            progress_dialog = QDialog(self)
            progress_dialog.setWindowTitle("正在计算")
            progress_dialog.setFixedSize(300, 100)

            dialog_layout = QVBoxLayout(progress_dialog)

            info_label = QLabel("正在计算用户相似度...", progress_dialog)
            dialog_layout.addWidget(info_label)

            progress_bar = QProgressBar(progress_dialog)
            progress_bar.setRange(0, 0)  # 设置为不确定模式
            dialog_layout.addWidget(progress_bar)

            progress_dialog.show()
            QApplication.processEvents()

            # 计算相似度
            self.similarity_matrix = self.recommender.calculate_user_similarity()

            # 关闭进度对话框
            progress_dialog.close()

            # 可视化相似度矩阵
            self.visualize_similarity_matrix()

            # 启用推荐按钮
            self.recommend_btn.setEnabled(True)

            # 提示成功
            QMessageBox.information(self, "成功", "用户相似度计算完成！")

        except Exception as e:
            logging.error(f"计算用户相似度时发生错误: {str(e)}")
            QMessageBox.critical(self, "错误", f"计算用户相似度时发生错误:\n{str(e)}")

    @error_handler
    def visualize_similarity_matrix(self):
        """
        可视化用户相似度矩阵
        """
        try:
            if self.similarity_matrix is None:
                return

            # 如果矩阵太大，只显示一部分
            max_display = 10
            if len(self.similarity_matrix) > max_display:
                # 取前max_display行和列
                display_matrix = self.similarity_matrix.iloc[:max_display, :max_display]
            else:
                display_matrix = self.similarity_matrix

            # 转换为列表以供热力图使用
            matrix_data = display_matrix.values.tolist()
            row_labels = display_matrix.index.tolist()
            col_labels = display_matrix.columns.tolist()

            # 在热力图中显示
            self.similarity_chart.set_title("用户相似度矩阵")
            self.similarity_chart.set_heatmap_data(matrix_data, row_labels, col_labels)

        except Exception as e:
            logging.error(f"可视化相似度矩阵时发生错误: {str(e)}")

    @error_handler
    def generate_recommendations(self):
        """
        生成基于用户的推荐并显示过程和结果
        """
        try:
            # 获取参数
            user_id = int(self.user_combo.currentText())
            n_neighbors = self.neighbors_spin.value()
            n_recommendations = self.recommendations_spin.value()

            # 清空之前的结果
            self.process_text.clear()
            self.results_table.setRowCount(0)

            # 添加过程开始信息
            self.process_text.setHtml(f"""
            <style>
            h3, h4 {{
                color: #0078D4;
                margin-top: 16px;
                margin-bottom: 8px;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin-bottom: 16px;
            }}
            th, td {{
                padding: 8px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: rgba(0, 122, 204, 0.2);
            }}
            hr {{
                border: none;
                border-top: 1px solid #ddd;
                margin: 16px 0;
            }}
            .formula {{
                background-color: rgba(0, 122, 204, 0.1);
                padding: 10px;
                border-radius: 4px;
                font-family: monospace;
                text-align: center;
                margin: 10px 0;
            }}
            </style>

            <h3>为用户 {user_id} 生成推荐</h3>
            <p>使用 {n_neighbors} 个最相似的用户生成 {n_recommendations} 条推荐</p>
            <hr>
            """)
            QApplication.processEvents()

            # 获取用户相似度
            if self.similarity_matrix is None:
                raise ValueError("尚未计算用户相似度")

            user_similarities = self.similarity_matrix.loc[user_id].sort_values(ascending=False)
            similar_users = user_similarities.index[1:n_neighbors + 1].tolist()  # 排除自己

            # 显示相似用户
            similar_users_html = f"""
            <h4>相似用户</h4>
            <table>
                <tr>
                    <th>用户ID</th>
                    <th>相似度</th>
                </tr>
            """

            for sim_user in similar_users:
                similarity = user_similarities[sim_user]
                similar_users_html += f"<tr><td>{sim_user}</td><td>{similarity:.4f}</td></tr>"

            similar_users_html += "</table><hr>"
            self.process_text.insertHtml(similar_users_html)
            QApplication.processEvents()

            # 获取用户已评分的图书
            user_item_matrix = self.recommender.create_user_item_matrix()
            user_ratings = user_item_matrix.loc[user_id]
            rated_books = user_ratings[user_ratings > 0]

            # 显示用户已评分的图书
            rated_books_html = f"""
            <h4>用户已评分的图书</h4>
            <table>
                <tr>
                    <th>图书ID</th>
                    <th>图书标题</th>
                    <th>评分</th>
                </tr>
            """

            for book_id, rating in rated_books.items():
                # 获取图书标题
                book_title = "未知"
                if self.recommender.books_df is not None:
                    title_series = self.recommender.books_df.loc[
                        self.recommender.books_df['book_id'] == book_id, 'title']
                    if not title_series.empty:
                        book_title = title_series.iloc[0]

                rated_books_html += f"<tr><td>{book_id}</td><td>{book_title}</td><td>{rating:.1f}</td></tr>"

            rated_books_html += "</table><hr>"
            self.process_text.insertHtml(rated_books_html)
            QApplication.processEvents()

            # 执行推荐
            self.process_text.insertHtml("<h4>正在生成推荐...</h4>")
            QApplication.processEvents()

            self.recommendations = self.recommender.get_user_based_recommendations(
                user_id=user_id,
                n_neighbors=n_neighbors,
                n_recommendations=n_recommendations
            )

            # 显示推荐过程的解释
            explanation_html = """
            <h4>推荐过程</h4>
            <ol>
                <li>找到与当前用户最相似的用户</li>
                <li>查看这些相似用户对用户未评分图书的评分</li>
                <li>根据相似度加权计算预测评分</li>
                <li>选择预测评分最高的图书进行推荐</li>
            </ol>

            <h4>计算公式</h4>
            <div class="formula">
                预测评分(u,i) = ∑(相似度(u,v) × 评分(v,i)) / ∑相似度(u,v)
            </div>
            <p>其中:</p>
            <ul>
                <li>u: 目标用户</li>
                <li>v: 相似用户</li>
                <li>i: 图书</li>
            </ul>
            <hr>
            """
            self.process_text.insertHtml(explanation_html)
            QApplication.processEvents()

            # 显示推荐结果
            self.display_recommendations()

            # 启用保存按钮
            self.save_btn.setEnabled(True)

        except Exception as e:
            logging.error(f"生成推荐时发生错误: {str(e)}")
            QMessageBox.critical(self, "错误", f"生成推荐时发生错误:\n{str(e)}")

    @error_handler
    def display_recommendations(self):
        """
        在表格中显示推荐结果
        """
        try:
            if not self.recommendations:
                return

            # 清空表格
            self.results_table.setRowCount(0)

            # 添加推荐结果
            for i, (book_id, book_title, score) in enumerate(self.recommendations):
                self.results_table.insertRow(i)

                # 添加图书ID
                id_item = QTableWidgetItem(str(book_id))
                id_item.setTextAlignment(Qt.AlignCenter)
                self.results_table.setItem(i, 0, id_item)

                # 添加图书标题
                title_item = QTableWidgetItem(book_title)
                self.results_table.setItem(i, 1, title_item)

                # 添加预测评分
                score_item = QTableWidgetItem(f"{score:.2f}")
                score_item.setTextAlignment(Qt.AlignCenter)

                # 根据评分设置背景颜色
                if score >= 4.5:
                    score_item.setBackground(QColor(108, 198, 68, 100))  # 绿色
                elif score >= 4.0:
                    score_item.setBackground(QColor(0, 122, 204, 100))  # 蓝色
                elif score >= 3.5:
                    score_item.setBackground(QColor(225, 151, 76, 100))  # 橙色

                self.results_table.setItem(i, 2, score_item)

            # 调整列宽
            self.results_table.resizeColumnsToContents()

        except Exception as e:
            logging.error(f"显示推荐结果时发生错误: {str(e)}")

    @error_handler
    def save_recommendations(self):
        """
        保存推荐结果到CSV文件
        """
        try:
            if not self.recommendations:
                QMessageBox.warning(self, "警告", "没有推荐结果可保存")
                return

            # 选择保存文件
            file_path, _ = QFileDialog.getSaveFileName(
                self, "保存推荐结果", "", "CSV文件 (*.csv)"
            )

            if not file_path:
                return

            # 保存推荐结果
            self.recommender.save_recommendations(self.recommendations, file_path)

            # 提示成功
            QMessageBox.information(self, "成功", f"推荐结果已保存到:\n{file_path}")

        except Exception as e:
            logging.error(f"保存推荐结果时发生错误: {str(e)}")
            QMessageBox.critical(self, "错误", f"保存推荐结果时发生错误:\n{str(e)}")


# 改进的基于物品的推荐界面
class ItemBasedRecommendationWidget(QWidget):
    """
    基于物品的推荐窗口
    """

    def __init__(self, recommender, theme, icons, parent=None):
        """
        初始化窗口

        参数:
            recommender: LibraryRecommender实例
            theme: 主题对象
            icons: 图标字典
            parent: 父窗口
        """
        super(ItemBasedRecommendationWidget, self).__init__(parent)
        self.recommender = recommender
        self.theme = theme
        self.icons = icons
        self.similarity_matrix = None
        self.recommendations = None
        self.init_ui()

    def init_ui(self):
        """
        初始化UI组件
        """
        # 创建主布局
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(20)

        # 顶部操作栏
        top_bar = QHBoxLayout()

        # 标题
        title_label = QLabel("基于物品的协同过滤推荐")
        font = title_label.font()
        font.setPointSize(16)
        font.setBold(True)
        title_label.setFont(font)
        top_bar.addWidget(title_label)

        top_bar.addStretch()

        # 操作按钮
        self.calc_similarity_btn = VSCodeButton("计算图书相似度", self.icons['book'])
        self.calc_similarity_btn.clicked.connect(self.calculate_similarity)
        top_bar.addWidget(self.calc_similarity_btn)

        main_layout.addLayout(top_bar)

        # 上半部分：控制面板和相似度矩阵可视化
        top_section = QSplitter(Qt.Horizontal)

        # 控制面板
        control_card = CardWidget("推荐参数设置")
        control_layout = QVBoxLayout()

        # 参数设置
        params_group = QWidget()
        params_layout = QGridLayout(params_group)
        params_layout.setColumnStretch(1, 1)  # 设置第二列可伸展

        # 用户选择
        self.user_label = QLabel("选择用户:")
        self.user_combo = QComboBox()

        # 参数设置
        self.recommendations_label = QLabel("推荐数量:")
        self.recommendations_spin = QSpinBox()
        self.recommendations_spin.setRange(1, 20)
        self.recommendations_spin.setValue(10)
        self.recommendations_spin.setToolTip("要生成的推荐图书数量")

        # 添加到布局
        params_layout.addWidget(self.user_label, 0, 0)
        params_layout.addWidget(self.user_combo, 0, 1)

        params_layout.addWidget(self.recommendations_label, 1, 0)
        params_layout.addWidget(self.recommendations_spin, 1, 1)

        control_layout.addWidget(params_group)

        # 添加分隔线
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        control_layout.addWidget(separator)

        # 操作按钮区
        buttons_layout = QVBoxLayout()

        self.recommend_btn = VSCodeButton("生成推荐", self.icons['book'])
        self.recommend_btn.clicked.connect(self.generate_recommendations)
        self.recommend_btn.setEnabled(False)
        buttons_layout.addWidget(self.recommend_btn)

        self.save_btn = VSCodeButton("保存推荐结果", None)
        self.save_btn.clicked.connect(self.save_recommendations)
        self.save_btn.setEnabled(False)
        buttons_layout.addWidget(self.save_btn)

        control_layout.addLayout(buttons_layout)

        # 添加伸缩器
        control_layout.addStretch()

        control_card.add_layout(control_layout)
        top_section.addWidget(control_card)

        # 相似度矩阵可视化
        similarity_card = CardWidget("图书相似度")
        similarity_layout = QVBoxLayout()

        # 显示相似度矩阵有点问题，因为图书可能很多，只显示热图可能不太直观
        # 这里添加一个筛选功能，允许用户选择特定图书查看其与其他图书的相似度
        filter_layout = QHBoxLayout()
        self.book_filter_label = QLabel("选择图书:")
        self.book_filter_combo = QComboBox()
        self.book_filter_combo.setEnabled(False)
        self.book_filter_combo.currentIndexChanged.connect(lambda _: self.update_book_similarity_chart())

        filter_layout.addWidget(self.book_filter_label)
        filter_layout.addWidget(self.book_filter_combo)

        similarity_layout.addLayout(filter_layout)

        self.similarity_chart = ModernChartWidget()
        similarity_layout.addWidget(self.similarity_chart)

        similarity_card.add_layout(similarity_layout)
        top_section.addWidget(similarity_card)

        # 设置分割比例
        top_section.setSizes([300, 700])

        main_layout.addWidget(top_section, 1)  # 1是伸展因子

        # 下半部分：推荐过程可视化和结果
        bottom_section = QSplitter(Qt.Horizontal)

        # 推荐过程可视化
        process_card = CardWidget("推荐过程")
        process_layout = QVBoxLayout()

        self.process_text = QTextEdit()
        self.process_text.setReadOnly(True)

        process_layout.addWidget(self.process_text)

        process_card.add_layout(process_layout)
        bottom_section.addWidget(process_card)

        # 推荐结果
        results_card = CardWidget("推荐结果")
        results_layout = QVBoxLayout()

        self.results_table = QTableWidget()
        self.results_table.setColumnCount(3)
        self.results_table.setHorizontalHeaderLabels(["图书ID", "图书标题", "预测评分"])
        self.results_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)

        results_layout.addWidget(self.results_table)

        results_card.add_layout(results_layout)
        bottom_section.addWidget(results_card)

        # 设置分割比例
        bottom_section.setSizes([500, 500])

        main_layout.addWidget(bottom_section, 1)  # 1是伸展因子

    def update_user_list(self):
        """
        更新用户下拉列表
        """
        try:
            self.user_combo.clear()

            if self.recommender.users_df is not None:
                for user_id in sorted(self.recommender.users_df['user_id'].unique()):
                    self.user_combo.addItem(str(user_id))
        except Exception as e:
            logging.error(f"更新用户列表时发生错误: {str(e)}")
            QMessageBox.critical(self, "错误", f"更新用户列表时发生错误:\n{str(e)}")

    def update_book_list(self):
        """
        更新图书下拉列表
        """
        try:
            self.book_filter_combo.clear()

            if self.recommender.books_df is not None:
                for _, row in self.recommender.books_df.iterrows():
                    self.book_filter_combo.addItem(f"{row['book_id']}: {row['title'][:30]}", row['book_id'])
        except Exception as e:
            logging.error(f"更新图书列表时发生错误: {str(e)}")
            QMessageBox.critical(self, "错误", f"更新图书列表时发生错误:\n{str(e)}")

    @error_handler
    def calculate_similarity(self):
        """
        计算图书相似度并可视化
        """
        try:
            # 显示进度对话框
            progress_dialog = QDialog(self)
            progress_dialog.setWindowTitle("正在计算")
            progress_dialog.setFixedSize(300, 100)

            dialog_layout = QVBoxLayout(progress_dialog)

            info_label = QLabel("正在计算图书相似度...", progress_dialog)
            dialog_layout.addWidget(info_label)

            progress_bar = QProgressBar(progress_dialog)
            progress_bar.setRange(0, 0)  # 设置为不确定模式
            dialog_layout.addWidget(progress_bar)

            progress_dialog.show()
            QApplication.processEvents()

            # 计算相似度
            self.similarity_matrix = self.recommender.calculate_book_similarity()

            # 关闭进度对话框
            progress_dialog.close()

            # 更新图书列表
            self.update_book_list()
            self.book_filter_combo.setEnabled(True)

            # 启用推荐按钮
            self.recommend_btn.setEnabled(True)

            # 提示成功
            QMessageBox.information(self, "成功", "图书相似度计算完成！")

        except Exception as e:
            logging.error(f"计算图书相似度时发生错误: {str(e)}")
            QMessageBox.critical(self, "错误", f"计算图书相似度时发生错误:\n{str(e)}")

    @error_handler
    def update_book_similarity_chart(self):
        """
        更新图书相似度图表，显示所选图书与其他图书的相似度
        """
        try:
            if self.similarity_matrix is None or self.book_filter_combo.count() == 0:
                return

            # 获取所选图书ID
            book_id = self.book_filter_combo.currentData()
            if book_id is None:
                return

            # 获取与所选图书的相似度
            if book_id in self.similarity_matrix.index:
                similarities = self.similarity_matrix.loc[book_id].sort_values(ascending=False)

                # 排除自身，取前10个最相似的图书
                similarities = similarities.iloc[1:11]

                # 获取图书标题
                book_titles = {}
                for b_id in similarities.index:
                    title_series = self.recommender.books_df.loc[
                        self.recommender.books_df['book_id'] == b_id, 'title']
                    if not title_series.empty:
                        book_titles[b_id] = title_series.iloc[0][:15]  # 限制标题长度
                    else:
                        book_titles[b_id] = f"Book {b_id}"

                # 准备图表数据
                book_ids = similarities.index.tolist()
                similarity_values = similarities.values.tolist()
                book_labels = [f"{book_titles.get(b_id, b_id)}" for b_id in book_ids]

                # 绘制条形图
                self.similarity_chart.set_title(f"与图书 {book_id} 最相似的图书")
                self.similarity_chart.set_labels("图书", "相似度")
                self.similarity_chart.set_bar_data(book_labels, similarity_values)

        except Exception as e:
            logging.error(f"更新图书相似度图表时发生错误: {str(e)}")

    @error_handler
    def generate_recommendations(self):
        """
        生成基于物品的推荐并显示过程和结果
        """
        try:
            # 获取参数
            user_id = int(self.user_combo.currentText())
            n_recommendations = self.recommendations_spin.value()

            # 清空之前的结果
            self.process_text.clear()
            self.results_table.setRowCount(0)

            # 添加过程开始信息
            self.process_text.setHtml(f"""
            <style>
            h3, h4 {{
                color: #0078D4;
                margin-top: 16px;
                margin-bottom: 8px;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin-bottom: 16px;
            }}
            th, td {{
                padding: 8px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: rgba(0, 122, 204, 0.2);
            }}
            hr {{
                border: none;
                border-top: 1px solid #ddd;
                margin: 16px 0;
            }}
            .formula {{
                background-color: rgba(0, 122, 204, 0.1);
                padding: 10px;
                border-radius: 4px;
                font-family: monospace;
                text-align: center;
                margin: 10px 0;
            }}
            </style>

            <h3>为用户 {user_id} 生成基于物品的推荐</h3>
            <p>生成 {n_recommendations} 条推荐</p>
            <hr>
            """)
            QApplication.processEvents()

            # 确保已计算图书相似度
            if self.similarity_matrix is None:
                raise ValueError("尚未计算图书相似度")

            # 获取用户已评分的图书
            user_item_matrix = self.recommender.create_user_item_matrix()
            if user_id not in user_item_matrix.index:
                raise ValueError(f"用户 {user_id} 没有评分记录")

            user_ratings = user_item_matrix.loc[user_id]
            rated_books = user_ratings[user_ratings > 0]

            # 显示用户已评分的图书
            rated_books_html = f"""
            <h4>用户已评分的图书</h4>
            <table>
                <tr>
                    <th>图书ID</th>
                    <th>图书标题</th>
                    <th>评分</th>
                </tr>
            """

            for book_id, rating in rated_books.items():
                # 获取图书标题
                book_title = "未知"
                if self.recommender.books_df is not None:
                    title_series = self.recommender.books_df.loc[
                        self.recommender.books_df['book_id'] == book_id, 'title']
                    if not title_series.empty:
                        book_title = title_series.iloc[0]

                rated_books_html += f"<tr><td>{book_id}</td><td>{book_title}</td><td>{rating:.1f}</td></tr>"

            rated_books_html += "</table><hr>"
            self.process_text.insertHtml(rated_books_html)
            QApplication.processEvents()

            # 显示基于物品的推荐过程的解释
            explanation_html = """
            <h4>基于物品的推荐过程</h4>
            <ol>
                <li>找到用户已评分的图书</li>
                <li>对于每本未评分的图书，计算与已评分图书的相似度</li>
                <li>基于相似度和用户对相似图书的评分，预测用户对未评分图书的评分</li>
                <li>推荐预测评分最高的图书</li>
            </ol>

            <h4>计算公式</h4>
            <div class="formula">
                预测评分(u,i) = ∑(相似度(i,j) × 评分(u,j)) / ∑相似度(i,j)
            </div>
            <p>其中:</p>
            <ul>
                <li>u: 用户</li>
                <li>i: 待预测评分的图书</li>
                <li>j: 用户已评分的图书</li>
            </ul>
            <hr>
            """
            self.process_text.insertHtml(explanation_html)
            QApplication.processEvents()

            # 执行推荐
            self.process_text.insertHtml("<h4>正在生成推荐...</h4>")
            QApplication.processEvents()

            self.recommendations = self.recommender.get_item_based_recommendations(
                user_id=user_id,
                n_recommendations=n_recommendations
            )

            # 显示推荐结果
            self.display_recommendations()

            # 启用保存按钮
            self.save_btn.setEnabled(True)

        except Exception as e:
            logging.error(f"生成基于物品的推荐时发生错误: {str(e)}")
            QMessageBox.critical(self, "错误", f"生成基于物品的推荐时发生错误:\n{str(e)}")

    @error_handler
    def display_recommendations(self):
        """
        在表格中显示推荐结果
        """
        try:
            if not self.recommendations:
                return

            # 清空表格
            self.results_table.setRowCount(0)

            # 添加推荐结果
            for i, (book_id, book_title, score) in enumerate(self.recommendations):
                self.results_table.insertRow(i)

                # 添加图书ID
                id_item = QTableWidgetItem(str(book_id))
                id_item.setTextAlignment(Qt.AlignCenter)
                self.results_table.setItem(i, 0, id_item)

                # 添加图书标题
                title_item = QTableWidgetItem(book_title)
                self.results_table.setItem(i, 1, title_item)

                # 添加预测评分
                score_item = QTableWidgetItem(f"{score:.2f}")
                score_item.setTextAlignment(Qt.AlignCenter)

                # 根据评分设置背景颜色
                if score >= 4.5:
                    score_item.setBackground(QColor(108, 198, 68, 100))  # 绿色
                elif score >= 4.0:
                    score_item.setBackground(QColor(0, 122, 204, 100))  # 蓝色
                elif score >= 3.5:
                    score_item.setBackground(QColor(225, 151, 76, 100))  # 橙色

                self.results_table.setItem(i, 2, score_item)

            # 调整列宽
            self.results_table.resizeColumnsToContents()

        except Exception as e:
            logging.error(f"显示推荐结果时发生错误: {str(e)}")

    @error_handler
    def save_recommendations(self):
        """
        保存推荐结果到CSV文件
        """
        try:
            if not self.recommendations:
                QMessageBox.warning(self, "警告", "没有推荐结果可保存")
                return

            # 选择保存文件
            file_path, _ = QFileDialog.getSaveFileName(
                self, "保存推荐结果", "", "CSV文件 (*.csv)"
            )

            if not file_path:
                return

            # 保存推荐结果
            self.recommender.save_recommendations(self.recommendations, file_path)

            # 提示成功
            QMessageBox.information(self, "成功", f"推荐结果已保存到:\n{file_path}")

        except Exception as e:
            logging.error(f"保存推荐结果时发生错误: {str(e)}")
            QMessageBox.critical(self, "错误", f"保存推荐结果时发生错误:\n{str(e)}")


# 改进的评估窗口
class EvaluationWidget(QWidget):
    """
    推荐系统评估窗口
    """

    def __init__(self, recommender, theme, icons, parent=None):
        """
        初始化窗口

        参数:
            recommender: LibraryRecommender实例
            theme: 主题对象
            icons: 图标字典
            parent: 父窗口
        """
        super(EvaluationWidget, self).__init__(parent)
        self.recommender = recommender
        self.theme = theme
        self.icons = icons
        self.init_ui()

    def init_ui(self):
        """
        初始化UI组件
        """
        # 创建主布局
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(20)

        # 顶部操作栏
        top_bar = QHBoxLayout()

        # 标题
        title_label = QLabel("推荐系统评估")
        font = title_label.font()
        font.setPointSize(16)
        font.setBold(True)
        title_label.setFont(font)
        top_bar.addWidget(title_label)

        top_bar.addStretch()

        main_layout.addLayout(top_bar)

        # 评估控制和结果区域
        eval_section = QSplitter(Qt.Vertical)

        # 控制面板
        control_card = CardWidget("评估参数设置")
        control_layout = QVBoxLayout()

        # 参数设置
        params_group = QWidget()
        params_layout = QGridLayout(params_group)
        params_layout.setColumnStretch(1, 1)  # 设置第二列可伸展

        # 测试集比例
        self.test_size_label = QLabel("测试集比例:")
        self.test_size_spin = QDoubleSpinBox()
        self.test_size_spin.setRange(0.1, 0.5)
        self.test_size_spin.setSingleStep(0.05)
        self.test_size_spin.setValue(0.2)
        self.test_size_spin.setToolTip("用于测试的数据比例")

        # 邻居数量
        self.neighbors_label = QLabel("用户邻居数量:")
        self.neighbors_spin = QSpinBox()
        self.neighbors_spin.setRange(1, 20)
        self.neighbors_spin.setValue(5)
        self.neighbors_spin.setToolTip("基于用户的推荐中考虑的相似用户数量")

        # 添加到布局
        params_layout.addWidget(self.test_size_label, 0, 0)
        params_layout.addWidget(self.test_size_spin, 0, 1)

        params_layout.addWidget(self.neighbors_label, 1, 0)
        params_layout.addWidget(self.neighbors_spin, 1, 1)

        control_layout.addWidget(params_group)

        # 评估选择
        evaluation_buttons = QHBoxLayout()

        self.evaluate_user_btn = VSCodeButton("评估基于用户的推荐", self.icons['user'])
        self.evaluate_user_btn.clicked.connect(lambda: self.evaluate_system('user_based'))
        evaluation_buttons.addWidget(self.evaluate_user_btn)

        self.evaluate_item_btn = VSCodeButton("评估基于物品的推荐", self.icons['book'])
        self.evaluate_item_btn.clicked.connect(lambda: self.evaluate_system('item_based'))
        evaluation_buttons.addWidget(self.evaluate_item_btn)

        control_layout.addLayout(evaluation_buttons)

        # 添加伸缩器
        control_layout.addStretch()

        control_card.add_layout(control_layout)
        eval_section.addWidget(control_card)

        # 结果显示
        results_splitter = QSplitter(Qt.Horizontal)

        # 结果文本区
        results_text_card = CardWidget("评估详情")
        results_text_layout = QVBoxLayout()

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)

        results_text_layout.addWidget(self.results_text)

        results_text_card.add_layout(results_text_layout)
        results_splitter.addWidget(results_text_card)

        # 结果可视化区
        results_chart_card = CardWidget("评估结果可视化")
        results_chart_layout = QVBoxLayout()

        self.results_chart = ModernChartWidget()

        results_chart_layout.addWidget(self.results_chart)

        results_chart_card.add_layout(results_chart_layout)
        results_splitter.addWidget(results_chart_card)

        # 设置分割比例
        results_splitter.setSizes([500, 500])

        eval_section.addWidget(results_splitter)

        # 设置分割比例
        eval_section.setSizes([200, 600])

        main_layout.addWidget(eval_section, 1)  # 1是伸展因子

        self.evaluate_matrix_btn = VSCodeButton("评估基于矩阵分解的推荐", self.icons['matrix'])
        self.evaluate_matrix_btn.clicked.connect(lambda: self.evaluate_system('matrix_factorization'))
        evaluation_buttons.addWidget(self.evaluate_matrix_btn)

        self.evaluate_hybrid_btn = VSCodeButton("评估混合推荐", self.icons['hybrid'])
        self.evaluate_hybrid_btn.clicked.connect(lambda: self.evaluate_system('hybrid'))
        evaluation_buttons.addWidget(self.evaluate_hybrid_btn)

    @error_handler
    def evaluate_system(self, method):
        """
        评估推荐系统性能

        参数:
            method: 推荐方法，'user_based'或'item_based'
        """
        try:
            # 获取参数
            test_size = self.test_size_spin.value()
            n_neighbors = self.neighbors_spin.value()

            # 清空之前的结果
            self.results_text.clear()

            # 添加评估开始信息
            self.results_text.setHtml(f"""
            <style>
            h3, h4 {{
                color: #0078D4;
                margin-top: 16px;
                margin-bottom: 8px;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin-bottom: 16px;
            }}
            th, td {{
                padding: 8px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: rgba(0, 122, 204, 0.2);
            }}
            hr {{
                border: none;
                border-top: 1px solid #ddd;
                margin: 16px 0;
            }}
            </style>

            <h3>正在评估{method}推荐系统</h3>
            <p><b>测试集比例</b>: {test_size}</p>
            """)

            if method == 'user_based':
                self.results_text.insertHtml(f"<p><b>用户邻居数量</b>: {n_neighbors}</p>")

            self.results_text.insertHtml("<hr>")
            QApplication.processEvents()

            # 创建测试集
            if self.recommender.ratings_df is None:
                raise ValueError("尚未加载评分数据")

            ratings = self.recommender.ratings_df
            np.random.seed(42)  # 固定随机种子，保证结果可重现
            test_indices = np.random.choice(ratings.index, size=int(len(ratings) * test_size), replace=False)
            test_data = ratings.loc[test_indices]

            self.results_text.insertHtml(f"""
            <p>创建了包含 <b>{len(test_data)}</b> 条评分的测试集 (总评分的 <b>{test_size:.1%}</b>)</p>
            <p>评估过程将对每条测试评分进行预测，并与实际评分进行比较。</p>
            <hr>
            """)
            QApplication.processEvents()

            # 显示进度对话框
            progress_dialog = QDialog(self)
            progress_dialog.setWindowTitle("正在评估")
            progress_dialog.setFixedSize(300, 100)

            dialog_layout = QVBoxLayout(progress_dialog)

            info_label = QLabel(f"正在评估{method}推荐系统...", progress_dialog)
            dialog_layout.addWidget(info_label)

            progress_bar = QProgressBar(progress_dialog)
            progress_bar.setRange(0, 0)  # 设置为不确定模式
            dialog_layout.addWidget(progress_bar)

            progress_dialog.show()
            QApplication.processEvents()

            # 执行评估
            if method == 'user_based':
                rmse = self.recommender.evaluate_recommendations(test_data, method='user_based',
                                                                 n_neighbors=n_neighbors)
                method_name = "基于用户的推荐"
            elif method == 'hybrid':
                # 获取混合推荐组件的方法和权重设置
                methods, weights = self.parent().hybrid_widget.get_selected_methods_and_weights()

                # 执行混合推荐评估
                rmse = self.recommender.evaluate_hybrid_recommendations(
                    test_data,
                    methods=methods,
                    weights=weights
                )
                method_name = "混合推荐"
            else:
                rmse = self.recommender.evaluate_recommendations(test_data, method='item_based')
                method_name = "基于物品的推荐"

            # 关闭进度对话框
            progress_dialog.close()

            # 显示评估结果
            if rmse is not None:
                result_color = "#6FCF97" if rmse < 1.0 else "#F2994A"  # 低RMSE使用绿色，高RMSE使用橙色

                self.results_text.insertHtml(f"""
                <h4>评估结果</h4>
                <div style="background-color: rgba(0, 122, 204, 0.1); padding: 15px; border-radius: 5px; margin: 15px 0;">
                    <p style="font-size: 16px; margin: 5px 0;">
                        <b>均方根误差 (RMSE)</b>: 
                        <span style="color: {result_color}; font-size: 20px; font-weight: bold;">{rmse:.4f}</span>
                    </p>
                    <p>RMSE越低表示预测越准确。通常小于1.0的RMSE被认为是较好的结果。</p>
                </div>

                <h4>评估说明</h4>
                <p>
                    均方根误差 (RMSE) 是评估推荐系统预测准确性的常用指标。它计算预测评分与实际评分之间的平方差的平均值的平方根。
                    RMSE值越小，表示预测越准确，模型性能越好。
                </p>
                """)

                # 绘制结果条形图
                self.plot_evaluation_results(method_name, rmse)
            else:
                self.results_text.insertHtml("""
                <p style="color: #E57373;">评估未完成，没有足够的数据。</p>
                <p>可能的原因:</p>
                <ul>
                    <li>测试集中的用户或图书在训练集中没有足够的评分记录</li>
                    <li>数据太稀疏，无法进行有效的预测</li>
                </ul>
                <p>建议尝试:</p>
                <ul>
                    <li>减小测试集比例</li>
                    <li>使用更密集的数据集</li>
                </ul>
                """)

        except Exception as e:
            logging.error(f"评估推荐系统时发生错误: {str(e)}")
            QMessageBox.critical(self, "错误", f"评估推荐系统时发生错误:\n{str(e)}")

    @error_handler
    def plot_evaluation_results(self, method, rmse):
        """
        绘制评估结果图表

        参数:
            method: 推荐方法名称
            rmse: 均方根误差
        """
        try:
            # 设置标题和轴标签
            self.results_chart.set_title(f"{method}评估结果")
            self.results_chart.set_labels("推荐方法", "均方根误差 (RMSE)")

            # 绘制条形图
            self.results_chart.set_bar_data([method], [rmse])

        except Exception as e:
            logging.error(f"绘制评估结果图表时发生错误: {str(e)}")


def create_matrix_icon():
    """创建矩阵分解图标"""
    matrix_img = QPixmap(64, 64)
    matrix_img.fill(Qt.transparent)
    painter = QPainter(matrix_img)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setPen(QPen(QColor("#2D9CDB"), 2))
    painter.setBrush(QBrush(QColor(45, 156, 219, 100)))

    # 绘制矩阵图形
    painter.drawRect(10, 10, 20, 20)
    painter.drawRect(34, 10, 20, 20)
    painter.drawRect(10, 34, 20, 20)
    painter.drawRect(34, 34, 20, 20)

    # 绘制链接线
    painter.drawLine(20, 20, 44, 44)
    painter.drawLine(20, 44, 44, 20)

    painter.end()
    return QIcon(matrix_img)


# 主窗口
class LibraryRecommenderUI(QMainWindow):
    """
    图书推荐系统主窗口
    """

    def __init__(self):
        """
        初始化主窗口
        """
        super(LibraryRecommenderUI, self).__init__()

        # 确保资源目录存在
        ensure_resources_dir()

        # 创建图标
        self.icons = create_simple_icons()
        self.icons['matrix'] = create_matrix_icon()
        self.icons['hybrid'] = create_hybrid_icon()

        # 创建深色主题
        self.theme = VSCodeTheme(is_dark=True)

        # 初始化推荐系统
        self.recommender = LibraryRecommender()

        # 初始化UI
        self.init_ui()

        # 设置窗口标题和大小
        self.setWindowTitle("图书馆推荐系统 - VSCode风格")
        self.resize(1200, 800)

    def init_ui(self):
        """
        初始化UI组件
        """
        # 首先创建状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("初始化中...")

        # 创建中央窗口部件
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # 主布局
        main_layout = QHBoxLayout(self.central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 创建侧边栏
        self.sidebar = Sidebar()
        main_layout.addWidget(self.sidebar)

        # 创建内容区
        self.content_stack = QStackedWidget()
        main_layout.addWidget(self.content_stack)

        # 设置布局比例 - 侧边栏占比较小
        main_layout.setStretch(0, 1)
        main_layout.setStretch(1, 5)

        # 添加侧边栏选项
        self.sidebar.add_item(self.icons['data'], "数据加载", "data_loading")
        self.sidebar.add_item(self.icons['user'], "基于用户的推荐", "user_based")
        self.sidebar.add_item(self.icons['book'], "基于物品的推荐", "item_based")
        self.sidebar.add_item(self.icons['evaluate'], "系统评估", "evaluation")
        self.sidebar.add_item(self.icons['settings'], "设置", "settings")

        # 创建内容页面
        self.load_widget = LoadingDataWidget(self.recommender, self.theme, self.icons)
        self.user_based_widget = UserBasedRecommendationWidget(self.recommender, self.theme, self.icons)
        self.item_based_widget = ItemBasedRecommendationWidget(self.recommender, self.theme, self.icons)
        self.evaluation_widget = EvaluationWidget(self.recommender, self.theme, self.icons)

        # 设置页面
        self.settings_widget = QWidget()
        settings_layout = QVBoxLayout(self.settings_widget)
        settings_layout.addWidget(QLabel("设置页面 - 待开发"))

        # 添加到内容栈
        self.content_stack.addWidget(self.load_widget)
        self.content_stack.addWidget(self.user_based_widget)
        self.content_stack.addWidget(self.item_based_widget)
        self.content_stack.addWidget(self.evaluation_widget)
        self.content_stack.addWidget(self.settings_widget)

        # 连接侧边栏信号
        self.sidebar.item_selected.connect(self.on_sidebar_item_selected)

        # 连接数据加载信号
        self.load_widget.data_loaded_signal.connect(self.on_data_loaded)

        # 初始只启用数据加载页面
        self.sidebar.items["user_based"].setEnabled(False)
        self.sidebar.items["item_based"].setEnabled(False)
        self.sidebar.items["evaluation"].setEnabled(False)

        # 创建菜单
        self.create_menus()

        # 默认选中数据加载页面
        self.sidebar.select_item("data_loading")

        # 更新状态栏
        self.status_bar.showMessage("就绪")

        # 在创建内容页面部分
        self.matrix_factorization_widget = MatrixFactorizationRecommendationWidget(self.recommender, self.theme,
                                                                                   self.icons)

        # 添加到内容栈
        self.content_stack.addWidget(self.matrix_factorization_widget)

        # 3. 添加侧边栏选项（在添加侧边栏选项部分）
        self.sidebar.add_item(self.icons['matrix'], "基于矩阵分解的推荐", "matrix_factorization")

        self.hybrid_widget = HybridRecommendationWidget(self.recommender, self.theme, self.icons)

        # 添加到内容栈
        self.content_stack.addWidget(self.hybrid_widget)

        # 添加侧边栏选项
        self.sidebar.add_item(self.icons['hybrid'], "混合推荐", "hybrid")

    def create_menus(self):
        """
        创建菜单栏
        """
        # 文件菜单
        file_menu = self.menuBar().addMenu("文件")

        # 创建示例数据
        create_demo_action = QAction("创建示例数据", self)
        create_demo_action.setShortcut(QKeySequence("Ctrl+N"))
        create_demo_action.triggered.connect(self.load_widget.create_demo_data)
        file_menu.addAction(create_demo_action)

        # 加载数据
        load_data_action = QAction("加载数据", self)
        load_data_action.setShortcut(QKeySequence("Ctrl+O"))
        load_data_action.triggered.connect(self.load_widget.load_data)
        file_menu.addAction(load_data_action)

        file_menu.addSeparator()

        # 退出
        exit_action = QAction("退出", self)
        exit_action.setShortcut(QKeySequence("Alt+F4"))
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # 视图菜单
        view_menu = self.menuBar().addMenu("视图")

        # 切换深色/浅色主题
        theme_action = QAction("切换深色/浅色主题", self)
        theme_action.setShortcut(QKeySequence("Ctrl+K Ctrl+T"))
        theme_action.triggered.connect(self.toggle_theme)
        view_menu.addAction(theme_action)

        # 帮助菜单
        help_menu = self.menuBar().addMenu("帮助")

        # 关于
        about_action = QAction("关于", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def on_sidebar_item_selected(self, item_id):
        """
        处理侧边栏项目选择事件

        参数:
            item_id (str): 选中的项目ID
        """
        # 切换到相应页面
        if item_id == "data_loading":
            self.content_stack.setCurrentWidget(self.load_widget)
            self.status_bar.showMessage("数据加载页面")
        elif item_id == "user_based":
            self.content_stack.setCurrentWidget(self.user_based_widget)
            self.status_bar.showMessage("基于用户的推荐页面")
        elif item_id == "item_based":
            self.content_stack.setCurrentWidget(self.item_based_widget)
            self.status_bar.showMessage("基于物品的推荐页面")
        elif item_id == "evaluation":
            self.content_stack.setCurrentWidget(self.evaluation_widget)
            self.status_bar.showMessage("系统评估页面")
        elif item_id == "settings":
            self.content_stack.setCurrentWidget(self.settings_widget)
            self.status_bar.showMessage("设置页面")
        elif item_id == "matrix_factorization":
            self.content_stack.setCurrentWidget(self.matrix_factorization_widget)
            self.status_bar.showMessage("基于矩阵分解的推荐页面")
        elif item_id == "hybrid":
            self.content_stack.setCurrentWidget(self.hybrid_widget)
            self.status_bar.showMessage("混合推荐页面")

    def on_data_loaded(self, success):
        """
        数据加载完成后的处理

        参数:
            success: 是否成功加载数据
        """
        if success:
            # 启用其他选项
            self.sidebar.items["user_based"].setEnabled(True)
            self.sidebar.items["item_based"].setEnabled(True)
            self.sidebar.items["evaluation"].setEnabled(True)

            # 更新用户列表
            self.user_based_widget.update_user_list()
            self.item_based_widget.update_user_list()

            self.sidebar.items["matrix_factorization"].setEnabled(True)
            self.matrix_factorization_widget.update_user_list()

            self.sidebar.items["hybrid"].setEnabled(True)
            self.hybrid_widget.update_user_list()

            # 更新状态栏
            self.status_bar.showMessage("数据加载成功")

    def toggle_theme(self):
        """
        切换深色/浅色主题
        """
        # 切换主题
        self.theme.is_dark = not self.theme.is_dark

        # 重新初始化主题
        self.theme.init_theme()

        # 应用到应用
        self.theme.apply_to_app(QApplication.instance())

        # 更新状态栏
        theme_name = "深色" if self.theme.is_dark else "浅色"
        self.status_bar.showMessage(f"已切换到{theme_name}主题")

    def show_about(self):
        """
        显示关于对话框
        """
        about_text = """
        <div style="text-align: center;">
            <h1 style="color: #0078D4;">图书馆推荐系统</h1>
            <p style="font-size: 14px;">版本: 1.0 VSCode风格</p>
            <p style="font-size: 14px;">日期: 2025-03-08</p>
            <p style="font-size: 14px;">作者: Claude</p>
            <hr style="margin: 10px 0;">
            <p>该系统使用协同过滤算法为用户推荐图书。</p>
            <p>支持基于用户和基于物品的推荐方法。</p>
            <div style="margin-top: 20px; padding: 10px; background-color: rgba(0, 122, 204, 0.1); border-radius: 5px;">
                <p style="margin: 5px 0;">使用VSCode风格的现代界面</p>
                <p style="margin: 5px 0;">提供直观的数据可视化</p>
                <p style="margin: 5px 0;">支持深色/浅色主题切换</p>
            </div>
        </div>
        """

        QMessageBox.about(self, "关于", about_text)


# 创建混合推荐图标（添加到create_simple_icons函数中）
def create_hybrid_icon():
    """创建混合推荐图标"""
    hybrid_img = QPixmap(64, 64)
    hybrid_img.fill(Qt.transparent)
    painter = QPainter(hybrid_img)
    painter.setRenderHint(QPainter.Antialiasing)

    # 绘制三种算法的组合表示
    # 用户图标（左上角）
    painter.setPen(QPen(QColor("#F2994A"), 2))
    painter.setBrush(QBrush(QColor(242, 153, 74, 100)))
    painter.drawEllipse(8, 8, 16, 16)  # 小用户头像

    # 书籍图标（右上角）
    painter.setPen(QPen(QColor("#6FCF97"), 2))
    painter.setBrush(QBrush(QColor(111, 207, 151, 100)))
    painter.drawRect(40, 8, 16, 20)  # 小书籍图标

    # 矩阵图标（下方）
    painter.setPen(QPen(QColor("#2D9CDB"), 2))
    painter.setBrush(QBrush(QColor(45, 156, 219, 100)))
    painter.drawRect(20, 36, 10, 10)
    painter.drawRect(34, 36, 10, 10)
    painter.drawRect(20, 50, 10, 10)
    painter.drawRect(34, 50, 10, 10)

    # 绘制连接线
    painter.setPen(QPen(QColor("#BB6BD9"), 2, Qt.DashLine))  # 紫色虚线
    painter.drawLine(16, 16, 32, 32)  # 连接用户到中心
    painter.drawLine(48, 16, 32, 32)  # 连接书籍到中心
    painter.drawLine(32, 45, 32, 32)  # 连接矩阵到中心

    # 绘制中心点
    painter.setPen(QPen(QColor("#BB6BD9"), 2))
    painter.setBrush(QBrush(QColor(187, 107, 217, 150)))
    painter.drawEllipse(28, 28, 8, 8)  # 中心点

    painter.end()
    return QIcon(hybrid_img)




def main():
    """
    主函数
    """
    try:
        # 初始化日志
        setup_logging()

        # 创建Qt应用
        app = QApplication(sys.argv)

        # 设置全局字体 - 避免使用弃用的构造函数
        # 直接使用系统字体
        app_font = QFont("Segoe UI", 9)  # Windows字体
        if sys.platform == "darwin":  # macOS
            app_font = QFont("SF Pro Text", 12)
        elif sys.platform.startswith("linux"):  # Linux
            app_font = QFont("Ubuntu", 10)

        app.setFont(app_font)

        # 创建主窗口
        main_window = LibraryRecommenderUI()

        # 应用主题
        main_window.theme.apply_to_app(app)

        # 显示窗口
        main_window.show()

        # 运行应用
        sys.exit(app.exec())

    except Exception as e:
        # 确保即使在图形界面启动前发生错误也能被记录和显示
        print(f"错误: {str(e)}")
        traceback.print_exc()

        # 如果已创建QApplication，则显示错误对话框
        if QApplication.instance():
            QMessageBox.critical(None, "严重错误", f"程序启动时发生错误:\n{str(e)}")

        sys.exit(1)

if __name__ == "__main__":
    main()