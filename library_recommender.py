#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import logging
import os
import sys
import traceback
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error


# 配置日志
def setup_logging():
    """
    设置日志系统，记录程序执行过程和错误信息
    """
    try:
        # 创建日志目录
        if not os.path.exists('logs'):
            os.makedirs('logs')

        # 配置日志格式和级别
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/library_recommender.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )

        logging.info("日志系统初始化成功")
    except Exception as e:
        print(f"日志系统初始化失败: {str(e)}")
        traceback.print_exc()
        sys.exit(1)


# 异常处理装饰器
def error_handler(func):
    """
    装饰器：捕获函数执行过程中的异常，记录错误信息并定位错误位置
    """

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # 获取当前异常的信息
            exc_type, exc_obj, exc_tb = sys.exc_info()

            # 获取发生异常的文件名和行号
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            line_no = exc_tb.tb_lineno

            # 记录详细的错误信息
            error_msg = f"错误类型: {exc_type.__name__}, 错误信息: {str(e)}, 文件: {fname}, 行号: {line_no}"
            logging.error(error_msg)

            # 打印完整的堆栈跟踪
            logging.error("详细堆栈信息:")
            for line in traceback.format_tb(exc_tb):
                logging.error(line.strip())

            # 如果是调试模式，重新抛出异常；否则返回None
            if os.environ.get('DEBUG', 'False').lower() == 'true':
                raise
            return None

    return wrapper


class LibraryRecommender:
    """
    图书馆推荐系统类
    使用协同过滤算法为用户推荐图书
    """

    def __init__(self, user_data_path=None, book_data_path=None, ratings_data_path=None):
        """
        初始化推荐系统

        参数:
            user_data_path (str): 用户数据文件路径
            book_data_path (str): 图书数据文件路径
            ratings_data_path (str): 评分数据文件路径
        """
        logging.info("初始化图书馆推荐系统")
        self.users_df = None
        self.books_df = None
        self.ratings_df = None
        self.user_similarity_matrix = None
        self.book_similarity_matrix = None

        # 矩阵分解相关属性
        self.user_factors = None
        self.item_factors = None
        self.user_index_to_id = None
        self.item_index_to_id = None
        self.user_id_to_index = None
        self.item_id_to_index = None

        # 如果提供了数据路径，则加载数据
        if user_data_path and book_data_path and ratings_data_path:
            self.load_data(user_data_path, book_data_path, ratings_data_path)

    @error_handler
    def load_data(self, user_data_path, book_data_path, ratings_data_path):
        """
        加载用户、图书和评分数据

        参数:
            user_data_path (str): 用户数据文件路径
            book_data_path (str): 图书数据文件路径
            ratings_data_path (str): 评分数据文件路径
        """
        logging.info(f"开始加载数据文件...")

        # 检查文件是否存在
        for path, desc in [(user_data_path, "用户数据"),
                           (book_data_path, "图书数据"),
                           (ratings_data_path, "评分数据")]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"{desc}文件不存在: {path}")

        try:
            # 加载用户数据
            logging.info(f"加载用户数据: {user_data_path}")
            self.users_df = pd.read_csv(user_data_path)
            logging.info(f"成功加载 {len(self.users_df)} 条用户记录")

            # 加载图书数据
            logging.info(f"加载图书数据: {book_data_path}")
            self.books_df = pd.read_csv(book_data_path)
            logging.info(f"成功加载 {len(self.books_df)} 条图书记录")

            # 加载评分数据
            logging.info(f"加载评分数据: {ratings_data_path}")
            self.ratings_df = pd.read_csv(ratings_data_path)
            logging.info(f"成功加载 {len(self.ratings_df)} 条评分记录")

            # 数据基本检查
            self._check_data_integrity()

        except pd.errors.ParserError as e:
            logging.error(f"CSV解析错误: {str(e)}")
            raise
        except Exception as e:
            logging.error(f"加载数据时发生错误: {str(e)}")
            raise

    @error_handler
    def _check_data_integrity(self):
        """
        检查数据完整性和一致性
        """
        logging.info("检查数据完整性...")

        # 检查用户数据
        if 'user_id' not in self.users_df.columns:
            raise ValueError("用户数据缺少'user_id'列")

        # 检查图书数据
        if 'book_id' not in self.books_df.columns:
            raise ValueError("图书数据缺少'book_id'列")

        # 检查评分数据
        required_cols = ['user_id', 'book_id', 'rating']
        missing_cols = [col for col in required_cols if col not in self.ratings_df.columns]
        if missing_cols:
            raise ValueError(f"评分数据缺少以下列: {', '.join(missing_cols)}")

        # 检查评分范围是否合理
        if not all(1 <= rating <= 5 for rating in self.ratings_df['rating']):
            logging.warning("评分数据中存在超出1-5范围的评分")

        # 检查参考完整性
        unknown_users = set(self.ratings_df['user_id']) - set(self.users_df['user_id'])
        if unknown_users:
            logging.warning(f"评分数据中存在未知用户ID: {unknown_users}")

        unknown_books = set(self.ratings_df['book_id']) - set(self.books_df['book_id'])
        if unknown_books:
            logging.warning(f"评分数据中存在未知图书ID: {unknown_books}")

        logging.info("数据完整性检查完成")

    @error_handler
    def create_user_item_matrix(self):
        """
        创建用户-物品矩阵
        行表示用户，列表示图书，单元格值表示评分
        处理重复的用户-图书评分

        返回:
            pandas.DataFrame: 用户-物品矩阵
        """
        logging.info("创建用户-物品矩阵...")

        if self.ratings_df is None:
            raise ValueError("尚未加载评分数据，无法创建用户-物品矩阵")

        try:
            # 检查是否存在重复的用户-图书评分
            duplicates = self.ratings_df.duplicated(subset=['user_id', 'book_id'], keep=False)

            if duplicates.any():
                duplicate_count = duplicates.sum()
                logging.warning(f"发现 {duplicate_count} 条重复的用户-图书评分记录，将使用平均值处理")

                # 处理方式：对于重复评分，取平均值
                ratings_df_clean = self.ratings_df.groupby(['user_id', 'book_id']).agg({
                    'rating': 'mean'  # 平均值策略
                    # 可选其他策略：'max'（最高评分）, 'min'（最低评分）, 'last'（最后一次评分）
                }).reset_index()

                # 创建用户-物品矩阵
                user_item_matrix = ratings_df_clean.pivot(
                    index='user_id',
                    columns='book_id',
                    values='rating'
                ).fillna(0)
            else:
                # 如果没有重复，直接创建矩阵
                user_item_matrix = self.ratings_df.pivot(
                    index='user_id',
                    columns='book_id',
                    values='rating'
                ).fillna(0)

            logging.info(f"成功创建用户-物品矩阵，形状: {user_item_matrix.shape}")
            return user_item_matrix

        except Exception as e:
            logging.error(f"创建用户-物品矩阵时发生错误: {str(e)}")
            raise

    @error_handler
    def calculate_user_similarity(self, similarity_metric='cosine'):
        """
        计算用户之间的相似度

        参数:
            similarity_metric (str): 相似度计算方法

        返回:
            pandas.DataFrame: 用户相似度矩阵
        """
        logging.info(f"计算用户相似度，使用{similarity_metric}方法...")

        try:
            # 获取用户-物品矩阵
            user_item_matrix = self.create_user_item_matrix()

            # 检查矩阵是否为空或只有一个用户
            if user_item_matrix.empty or len(user_item_matrix) <= 1:
                logging.warning("用户-物品矩阵为空或只有一个用户，无法计算相似度")
                return pd.DataFrame()

            # 检查矩阵是否包含任何NaN值，并用0填充
            if user_item_matrix.isnull().any().any():
                logging.warning("用户-物品矩阵包含NaN值，已用0填充")
                user_item_matrix = user_item_matrix.fillna(0)

            # 确保矩阵中至少有一个非零值
            if (user_item_matrix == 0).all().all():
                logging.warning("用户-物品矩阵全为0，无法计算有意义的相似度")
                return pd.DataFrame(
                    np.identity(len(user_item_matrix)),
                    index=user_item_matrix.index,
                    columns=user_item_matrix.index
                )

            if similarity_metric == 'cosine':
                # 使用余弦相似度
                similarity = cosine_similarity(user_item_matrix)
                self.user_similarity_matrix = pd.DataFrame(
                    similarity,
                    index=user_item_matrix.index,
                    columns=user_item_matrix.index
                )
                logging.info("用户相似度计算完成")
                return self.user_similarity_matrix
            else:
                raise ValueError(f"不支持的相似度计算方法: {similarity_metric}")

        except Exception as e:
            logging.error(f"计算用户相似度时发生错误: {str(e)}")
            raise

    @error_handler
    def calculate_book_similarity(self, similarity_metric='cosine'):
        """
        计算图书之间的相似度

        参数:
            similarity_metric (str): 相似度计算方法，目前支持'cosine'余弦相似度

        返回:
            pandas.DataFrame: 图书相似度矩阵
        """
        logging.info(f"计算图书相似度，使用{similarity_metric}方法...")

        try:
            # 获取用户-物品矩阵并转置
            user_item_matrix = self.create_user_item_matrix()

            # 检查矩阵是否为空
            if user_item_matrix.empty:
                logging.warning("用户-物品矩阵为空，无法计算图书相似度")
                return pd.DataFrame()

            book_user_matrix = user_item_matrix.transpose()

            # 检查矩阵是否为空或只有一本书
            if book_user_matrix.empty or len(book_user_matrix) <= 1:
                logging.warning("图书-用户矩阵为空或只有一本书，无法计算相似度")
                return pd.DataFrame()

            # 检查矩阵是否包含任何NaN值，并用0填充
            if book_user_matrix.isnull().any().any():
                logging.warning("图书-用户矩阵包含NaN值，已用0填充")
                book_user_matrix = book_user_matrix.fillna(0)

            # 确保矩阵中至少有一个非零值
            if (book_user_matrix == 0).all().all():
                logging.warning("图书-用户矩阵全为0，无法计算有意义的相似度")
                return pd.DataFrame(
                    np.identity(len(book_user_matrix)),
                    index=book_user_matrix.index,
                    columns=book_user_matrix.index
                )

            if similarity_metric == 'cosine':
                # 使用余弦相似度
                similarity = cosine_similarity(book_user_matrix)
                self.book_similarity_matrix = pd.DataFrame(
                    similarity,
                    index=book_user_matrix.index,
                    columns=book_user_matrix.index
                )
                logging.info("图书相似度计算完成")
                return self.book_similarity_matrix
            else:
                raise ValueError(f"不支持的相似度计算方法: {similarity_metric}")

        except Exception as e:
            logging.error(f"计算图书相似度时发生错误: {str(e)}")
            raise

    @error_handler
    def get_user_based_recommendations(self, user_id, n_neighbors=5, n_recommendations=10):
        """
        基于用户的协同过滤推荐

        参数:
            user_id (int): 用户ID
            n_neighbors (int): 考虑的相似用户数量
            n_recommendations (int): 推荐图书数量

        返回:
            list: 推荐图书ID和预测评分
        """
        logging.info(f"为用户 {user_id} 生成基于用户的推荐，使用 {n_neighbors} 个邻居...")

        try:
            # 检查用户是否存在
            if user_id not in self.users_df['user_id'].values:
                raise ValueError(f"用户ID {user_id} 不存在")

            # 确保已计算用户相似度
            if self.user_similarity_matrix is None:
                self.calculate_user_similarity()

            # 获取用户-物品矩阵
            user_item_matrix = self.create_user_item_matrix()

            # 获取用户已评分的图书
            user_ratings = user_item_matrix.loc[user_id]
            user_rated_books = user_ratings[user_ratings > 0].index.tolist()

            # 获取用户未评分的图书
            unrated_books = user_ratings[user_ratings == 0].index.tolist()

            # 获取最相似的n个用户
            user_similarities = self.user_similarity_matrix.loc[user_id].sort_values(ascending=False)
            similar_users = user_similarities.index[1:n_neighbors + 1].tolist()  # 排除自己

            logging.info(f"找到 {len(similar_users)} 个相似用户")

            # 为每本未评分的书计算预测评分
            predictions = {}
            for book_id in unrated_books:
                # 初始化分子和分母
                numerator = 0
                denominator = 0

                # 对每个相似用户进行计算
                for sim_user in similar_users:
                    # 获取相似用户对该书的评分
                    rating = user_item_matrix.loc[sim_user, book_id]

                    # 如果相似用户评价过这本书
                    if rating > 0:
                        # 获取相似度
                        similarity = self.user_similarity_matrix.loc[user_id, sim_user]

                        # 加权求和
                        numerator += similarity * rating
                        denominator += similarity

                # 计算预测评分
                if denominator > 0:
                    predictions[book_id] = numerator / denominator

            # 按预测评分排序并返回前n_recommendations本
            sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
            recommendations = sorted_predictions[:n_recommendations]

            logging.info(f"成功生成 {len(recommendations)} 条推荐")

            # 添加图书标题
            recommendations_with_titles = []
            for book_id, score in recommendations:
                book_title = self.books_df.loc[self.books_df['book_id'] == book_id, 'title'].values[0]
                recommendations_with_titles.append((book_id, book_title, score))

            return recommendations_with_titles

        except Exception as e:
            logging.error(f"生成基于用户的推荐时发生错误: {str(e)}")
            raise

    @error_handler
    def get_item_based_recommendations(self, user_id, n_recommendations=10):
        """
        基于物品的协同过滤推荐

        参数:
            user_id (int): 用户ID
            n_recommendations (int): 推荐图书数量

        返回:
            list: 推荐图书ID和预测评分
        """
        logging.info(f"为用户 {user_id} 生成基于物品的推荐...")

        try:
            # 检查用户是否存在
            if user_id not in self.users_df['user_id'].values:
                raise ValueError(f"用户ID {user_id} 不存在")

            # 确保已计算图书相似度
            if self.book_similarity_matrix is None:
                self.calculate_book_similarity()

            # 获取用户-物品矩阵
            user_item_matrix = self.create_user_item_matrix()

            # 获取用户已评分的图书
            user_ratings = user_item_matrix.loc[user_id]
            rated_books = user_ratings[user_ratings > 0]

            # 获取用户未评分的图书
            unrated_books = user_ratings[user_ratings == 0].index.tolist()

            # 为每本未评分的书计算预测评分
            predictions = {}
            for unrated_book in unrated_books:
                prediction = 0
                total_similarity = 0

                # 基于用户已评分的书计算加权平均
                for rated_book, rating in rated_books.items():
                    # 获取两本书的相似度
                    similarity = self.book_similarity_matrix.loc[rated_book, unrated_book]

                    if similarity > 0:  # 只考虑正相似度
                        prediction += similarity * rating
                        total_similarity += similarity

                # 如果有足够的相似度，计算预测评分
                if total_similarity > 0:
                    predictions[unrated_book] = prediction / total_similarity

            # 按预测评分排序并返回前n_recommendations本
            sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
            recommendations = sorted_predictions[:n_recommendations]

            logging.info(f"成功生成 {len(recommendations)} 条推荐")

            # 添加图书标题
            recommendations_with_titles = []
            for book_id, score in recommendations:
                book_title = self.books_df.loc[self.books_df['book_id'] == book_id, 'title'].values[0]
                recommendations_with_titles.append((book_id, book_title, score))

            return recommendations_with_titles

        except Exception as e:
            logging.error(f"生成基于物品的推荐时发生错误: {str(e)}")
            raise

    @error_handler
    def train_matrix_factorization(self, n_factors=20, learning_rate=0.005, regularization=0.02, n_epochs=100):
        """
        训练矩阵分解模型，使用随机梯度下降法

        参数:
            n_factors (int): 潜在因子数量
            learning_rate (float): 梯度下降学习率
            regularization (float): 正则化参数，防止过拟合
            n_epochs (int): 训练迭代次数

        返回:
            tuple: (P, Q) 用户和物品特征矩阵
        """
        logging.info(
            f"开始训练矩阵分解模型，参数: 因子数量={n_factors}, 学习率={learning_rate}, 正则化={regularization}, 迭代次数={n_epochs}")

        try:
            # 获取用户-物品矩阵
            user_item_matrix = self.create_user_item_matrix()
            R = user_item_matrix.values

            # 获取矩阵维度
            n_users, n_items = R.shape
            logging.info(f"用户-物品矩阵维度: {n_users}用户 x {n_items}物品")

            # 初始化用户和物品特征矩阵（使用小的随机值）
            np.random.seed(42)  # 设置随机种子，保证结果可重现
            P = np.random.normal(scale=0.1, size=(n_users, n_factors))
            Q = np.random.normal(scale=0.1, size=(n_items, n_factors))

            # 获取已有评分的索引
            user_indices, item_indices = R.nonzero()

            # 存储矩阵索引与用户/物品ID的映射关系
            self.user_index_to_id = {i: user_id for i, user_id in enumerate(user_item_matrix.index)}
            self.item_index_to_id = {i: item_id for i, item_id in enumerate(user_item_matrix.columns)}

            # 存储用户/物品ID与矩阵索引的映射关系
            self.user_id_to_index = {user_id: i for i, user_id in enumerate(user_item_matrix.index)}
            self.item_id_to_index = {item_id: i for i, item_id in enumerate(user_item_matrix.columns)}

            # 使用随机梯度下降训练模型
            for epoch in range(n_epochs):
                # 打乱训练顺序
                indices = np.arange(len(user_indices))
                np.random.shuffle(indices)

                # 遍历所有已有评分
                for idx in indices:
                    u = user_indices[idx]
                    i = item_indices[idx]

                    # 计算预测误差
                    prediction = np.dot(P[u], Q[i])
                    error = R[u, i] - prediction

                    # 更新用户和物品特征
                    P[u] += learning_rate * (error * Q[i] - regularization * P[u])
                    Q[i] += learning_rate * (error * P[u] - regularization * Q[i])

                # 计算当前轮次的RMSE（用于监控训练进度）
                predictions = np.sum(P[:, np.newaxis, :] * Q[np.newaxis, :, :], axis=2)
                mask = R > 0  # 只考虑已有评分
                rmse = np.sqrt(np.mean(np.power(R - predictions, 2) * mask))

                logging.info(f"轮次 {epoch + 1}/{n_epochs}, RMSE: {rmse:.4f}")

            # 存储训练好的矩阵
            self.user_factors = P
            self.item_factors = Q

            logging.info("矩阵分解模型训练完成")
            return P, Q

        except Exception as e:
            logging.error(f"训练矩阵分解模型时发生错误: {str(e)}")
            raise

    @error_handler
    def get_matrix_factorization_recommendations(self, user_id, n_recommendations=10):
        """
        使用矩阵分解模型生成推荐

        参数:
            user_id (int): 用户ID
            n_recommendations (int): 推荐数量

        返回:
            list: 推荐图书及预测评分
        """
        logging.info(f"为用户 {user_id} 生成基于矩阵分解的推荐...")

        try:
            # 检查模型是否已训练
            if not hasattr(self, 'user_factors') or self.user_factors is None:
                raise ValueError("矩阵分解模型尚未训练。请先调用train_matrix_factorization()方法")

            # 检查用户是否存在
            if user_id not in self.user_id_to_index:
                raise ValueError(f"用户ID {user_id} 在训练数据中不存在")

            # 获取用户索引
            user_idx = self.user_id_to_index[user_id]

            # 获取用户评分
            user_item_matrix = self.create_user_item_matrix()
            user_ratings = user_item_matrix.loc[user_id]

            # 获取未评分的物品
            unrated_items = user_ratings[user_ratings == 0].index.tolist()

            # 为未评分物品计算预测评分
            predictions = {}
            for item_id in unrated_items:
                if item_id in self.item_id_to_index:
                    item_idx = self.item_id_to_index[item_id]
                    # 使用潜在因子的点积计算预测评分
                    prediction = np.dot(self.user_factors[user_idx], self.item_factors[item_idx])
                    predictions[item_id] = prediction

            # 按预测评分排序
            sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
            recommendations = sorted_predictions[:n_recommendations]

            # 添加图书标题
            recommendations_with_titles = []
            for book_id, score in recommendations:
                book_title = self.books_df.loc[self.books_df['book_id'] == book_id, 'title'].values[0]
                recommendations_with_titles.append((book_id, book_title, score))

            logging.info(f"成功生成 {len(recommendations_with_titles)} 条矩阵分解推荐")
            return recommendations_with_titles

        except Exception as e:
            logging.error(f"生成矩阵分解推荐时发生错误: {str(e)}")
            raise

    @error_handler
    def evaluate_recommendations(self, test_data, method='user_based', n_neighbors=5, **kwargs):
        """
        评估推荐系统性能

        参数:
            test_data (pandas.DataFrame): 测试数据，包含user_id, book_id, rating列
            method (str): 评估方法，'user_based'、'item_based'或'matrix_factorization'
            n_neighbors (int): 基于用户方法中考虑的相似用户数量
            **kwargs: 其他参数，如matrix_factorization的超参数

        返回:
            float: 均方根误差(RMSE)
        """
        logging.info(f"评估{method}推荐方法的性能...")

        try:
            # 实际评分和预测评分列表
            actual_ratings = []
            predicted_ratings = []

            # 临时保存原始评分数据
            original_ratings = self.ratings_df.copy()

            # 如果是矩阵分解方法，预先训练模型
            if method == 'matrix_factorization':
                # 创建一个用于训练的临时数据集（排除测试数据）
                train_ratings = self.ratings_df.copy()
                for _, row in test_data.iterrows():
                    user_id = row['user_id']
                    book_id = row['book_id']
                    indices = train_ratings[(train_ratings['user_id'] == user_id) &
                                            (train_ratings['book_id'] == book_id)].index
                    if not indices.empty:
                        train_ratings = train_ratings.drop(indices)

                # 使用训练数据集
                self.ratings_df = train_ratings

                # 设置矩阵分解参数
                n_factors = kwargs.get('n_factors', 20)
                learning_rate = kwargs.get('learning_rate', 0.005)
                regularization = kwargs.get('regularization', 0.02)
                n_epochs = kwargs.get('n_epochs', 20)  # 评估时使用较少的轮次

                # 训练矩阵分解模型
                logging.info(
                    f"训练矩阵分解模型用于评估，参数: 因子数={n_factors}, 学习率={learning_rate}, 正则化={regularization}, 迭代={n_epochs}")
                self.train_matrix_factorization(
                    n_factors=n_factors,
                    learning_rate=learning_rate,
                    regularization=regularization,
                    n_epochs=n_epochs
                )

                # 为测试集中的每个用户-物品对计算预测评分
                for _, row in test_data.iterrows():
                    user_id = row['user_id']
                    book_id = row['book_id']
                    actual_rating = row['rating']

                    # 检查用户和物品是否在训练模型中
                    if (user_id in self.user_id_to_index and
                            book_id in self.item_id_to_index):
                        user_idx = self.user_id_to_index[user_id]
                        item_idx = self.item_id_to_index[book_id]

                        # 计算预测评分
                        predicted_rating = np.dot(self.user_factors[user_idx],
                                                  self.item_factors[item_idx])

                        # 确保预测评分在合理范围内
                        predicted_rating = max(1.0, min(5.0, predicted_rating))

                        # 添加到评估列表
                        actual_ratings.append(actual_rating)
                        predicted_ratings.append(predicted_rating)

            else:  # 基于用户或基于物品的方法
                # 对测试集中的每条评分进行预测
                for _, row in test_data.iterrows():
                    user_id = row['user_id']
                    book_id = row['book_id']
                    actual_rating = row['rating']

                    # 临时从训练数据中移除该评分
                    temp_ratings = self.ratings_df.copy()
                    temp_index = temp_ratings[(temp_ratings['user_id'] == user_id) &
                                              (temp_ratings['book_id'] == book_id)].index
                    if not temp_index.empty:
                        temp_ratings = temp_ratings.drop(temp_index)

                        # 使用临时数据
                        self.ratings_df = temp_ratings

                        # 基于方法选择预测方式
                        predicted_rating = 0
                        if method == 'user_based':
                            # 创建新的用户-物品矩阵
                            user_item_matrix = self.create_user_item_matrix()

                            # 重新计算用户相似度
                            self.calculate_user_similarity()

                            # 检查用户是否在矩阵中
                            if user_id in self.user_similarity_matrix.index:
                                # 获取最相似的用户
                                similar_users = self.user_similarity_matrix.loc[user_id].sort_values(
                                    ascending=False).index[
                                                1:n_neighbors + 1].tolist()

                                # 计算预测评分
                                numerator = 0
                                denominator = 0
                                for sim_user in similar_users:
                                    # 获取相似用户对该书的评分
                                    if sim_user in user_item_matrix.index and book_id in user_item_matrix.columns:
                                        rating = user_item_matrix.loc[sim_user, book_id]

                                        # 如果相似用户评价过这本书
                                        if rating > 0:
                                            # 获取相似度
                                            similarity = self.user_similarity_matrix.loc[user_id, sim_user]

                                            # 加权求和
                                            numerator += similarity * rating
                                            denominator += similarity

                                # 计算预测评分
                                if denominator > 0:
                                    predicted_rating = numerator / denominator

                        elif method == 'item_based':
                            # 创建新的用户-物品矩阵
                            user_item_matrix = self.create_user_item_matrix()

                            # 重新计算图书相似度
                            self.calculate_book_similarity()

                            # 获取用户已评分的图书
                            if user_id in user_item_matrix.index:
                                user_ratings = user_item_matrix.loc[user_id]
                                rated_books = user_ratings[user_ratings > 0]

                                # 基于用户已评分的书计算加权平均
                                prediction = 0
                                total_similarity = 0

                                for rated_book, rating in rated_books.items():
                                    # 检查书籍是否在相似度矩阵中
                                    if (rated_book in self.book_similarity_matrix.index and
                                            book_id in self.book_similarity_matrix.columns):
                                        similarity = self.book_similarity_matrix.loc[rated_book, book_id]

                                        if similarity > 0:  # 只考虑正相似度
                                            prediction += similarity * rating
                                            total_similarity += similarity

                                # 如果有足够的相似度，计算预测评分
                                if total_similarity > 0:
                                    predicted_rating = prediction / total_similarity

                        # 确保预测评分在合理范围内
                        if predicted_rating > 0:
                            predicted_rating = max(1.0, min(5.0, predicted_rating))

                            # 添加到评估列表
                            actual_ratings.append(actual_rating)
                            predicted_ratings.append(predicted_rating)

            # 恢复原始数据
            self.ratings_df = original_ratings

            # 计算RMSE
            if len(actual_ratings) > 0 and len(predicted_ratings) > 0:
                rmse = np.sqrt(mean_squared_error(actual_ratings, predicted_ratings))
                logging.info(f"{method}方法的RMSE为: {rmse:.4f} (基于 {len(actual_ratings)} 条预测)")
                return rmse
            else:
                logging.warning(f"没有足够的数据进行评估，方法: {method}")
                return None

        except Exception as e:
            logging.error(f"评估推荐系统性能时发生错误: {str(e)}")
            # 确保恢复原始数据
            if 'original_ratings' in locals():
                self.ratings_df = original_ratings
            raise

    @error_handler
    def data_summary(self):
        """
        生成数据摘要统计信息

        返回:
            dict: 数据统计摘要
        """
        logging.info("生成数据摘要统计...")

        summary = {}

        try:
            if self.users_df is not None:
                summary['users_count'] = len(self.users_df)

            if self.books_df is not None:
                summary['books_count'] = len(self.books_df)

            if self.ratings_df is not None:
                summary['ratings_count'] = len(self.ratings_df)
                summary['average_rating'] = self.ratings_df['rating'].mean()
                summary['rating_distribution'] = self.ratings_df['rating'].value_counts().to_dict()

                # 计算稀疏度
                if 'users_count' in summary and 'books_count' in summary:
                    total_possible_ratings = summary['users_count'] * summary['books_count']
                    summary['sparsity'] = 1.0 - (summary['ratings_count'] / total_possible_ratings)

                # 每用户平均评分数
                user_rating_counts = self.ratings_df['user_id'].value_counts()
                summary['ratings_per_user'] = {
                    'min': user_rating_counts.min(),
                    'max': user_rating_counts.max(),
                    'mean': user_rating_counts.mean()
                }

                # 每本书平均评分数
                book_rating_counts = self.ratings_df['book_id'].value_counts()
                summary['ratings_per_book'] = {
                    'min': book_rating_counts.min(),
                    'max': book_rating_counts.max(),
                    'mean': book_rating_counts.mean()
                }

            logging.info("数据摘要统计生成完成")
            return summary

        except Exception as e:
            logging.error(f"生成数据摘要时发生错误: {str(e)}")
            raise

    @error_handler
    def save_recommendations(self, recommendations, output_path):
        """
        保存推荐结果到文件

        参数:
            recommendations (list): 推荐结果列表
            output_path (str): 输出文件路径
        """
        logging.info(f"保存推荐结果到: {output_path}")

        try:
            # 将推荐结果转换为数据框
            if recommendations and isinstance(recommendations[0], tuple) and len(recommendations[0]) == 3:
                # 如果包含图书标题
                df = pd.DataFrame(recommendations, columns=['book_id', 'title', 'predicted_rating'])
            elif recommendations and isinstance(recommendations[0], tuple) and len(recommendations[0]) == 2:
                # 如果只有ID和评分
                df = pd.DataFrame(recommendations, columns=['book_id', 'predicted_rating'])
            else:
                raise ValueError("推荐结果格式不正确")

            # 保存到CSV
            df.to_csv(output_path, index=False, encoding='utf-8')
            logging.info(f"成功保存 {len(recommendations)} 条推荐记录")

        except Exception as e:
            logging.error(f"保存推荐结果时发生错误: {str(e)}")
            raise

    @error_handler
    def get_hybrid_recommendations(self, user_id, n_recommendations=10, methods=None, weights=None):
        """
        混合推荐方法，结合多种推荐算法的结果

        参数:
            user_id (int): 用户ID
            n_recommendations (int): 推荐数量
            methods (list): 要使用的推荐方法列表，可选值: 'user_based', 'item_based', 'matrix_factorization'
                            默认使用所有三种方法
            weights (list): 各方法的权重，需与methods列表长度相同
                            默认平均权重

        返回:
            list: 推荐图书及预测评分
        """
        logging.info(f"为用户 {user_id} 生成混合推荐...")

        try:
            # 默认使用所有方法
            if methods is None:
                methods = ['user_based', 'item_based', 'matrix_factorization']

            # 检查指定的方法是否有效
            valid_methods = ['user_based', 'item_based', 'matrix_factorization']
            for method in methods:
                if method not in valid_methods:
                    raise ValueError(f"无效的推荐方法: {method}。有效值为: {valid_methods}")

            # 检查用户是否存在
            if user_id not in self.users_df['user_id'].values:
                raise ValueError(f"用户ID {user_id} 不存在")

            # 如果未指定权重，则使用平均权重
            if weights is None:
                weights = [1.0 / len(methods)] * len(methods)

            # 检查权重和方法列表长度是否匹配
            if len(weights) != len(methods):
                raise ValueError(f"权重列表长度 ({len(weights)}) 与方法列表长度 ({len(methods)}) 不匹配")

            # 检查权重是否为正数且总和为1
            if any(w < 0 for w in weights) or abs(sum(weights) - 1.0) > 1e-10:
                logging.warning("权重应为正数且总和为1.0，已自动归一化")
                weights = [max(0, w) for w in weights]  # 确保权重为正
                total = sum(weights)
                if total > 0:
                    weights = [w / total for w in weights]  # 归一化
                else:
                    weights = [1.0 / len(methods)] * len(methods)  # 回退到平均权重

            # 生成各方法的推荐结果
            all_recommendations = {}

            # 获取用户-物品矩阵
            user_item_matrix = self.create_user_item_matrix()

            # 获取用户已评分的图书
            user_ratings = user_item_matrix.loc[user_id]
            rated_books = set(user_ratings[user_ratings > 0].index.tolist())

            # 为每种方法获取推荐
            for i, method in enumerate(methods):
                if method == 'user_based':
                    # 确保已计算用户相似度
                    if self.user_similarity_matrix is None:
                        self.calculate_user_similarity()

                    recs = self.get_user_based_recommendations(
                        user_id=user_id,
                        n_recommendations=n_recommendations * 2  # 获取更多推荐以便后续筛选
                    )

                elif method == 'item_based':
                    # 确保已计算图书相似度
                    if self.book_similarity_matrix is None:
                        self.calculate_book_similarity()

                    recs = self.get_item_based_recommendations(
                        user_id=user_id,
                        n_recommendations=n_recommendations * 2
                    )

                elif method == 'matrix_factorization':
                    # 确保矩阵分解模型已训练
                    if not hasattr(self, 'user_factors') or self.user_factors is None:
                        logging.info("矩阵分解模型尚未训练，正在使用默认参数训练...")
                        self.train_matrix_factorization()

                    recs = self.get_matrix_factorization_recommendations(
                        user_id=user_id,
                        n_recommendations=n_recommendations * 2
                    )

                # 添加到总推荐字典，标准化为(book_id, score)格式
                weight = weights[i]
                for book_id, title, score in recs:
                    if book_id in all_recommendations:
                        all_recommendations[book_id] = (all_recommendations[book_id][0],
                                                        all_recommendations[book_id][1] + score * weight)
                    else:
                        all_recommendations[book_id] = (title, score * weight)

            # 如果没有任何推荐结果
            if not all_recommendations:
                logging.warning(f"没有生成任何推荐结果，用户ID: {user_id}")
                return []

            # 按照加权得分排序并返回前n_recommendations个推荐
            sorted_recs = sorted(
                [(book_id, title, score) for book_id, (title, score) in all_recommendations.items()],
                key=lambda x: x[2],
                reverse=True
            )

            recommendations = sorted_recs[:n_recommendations]

            logging.info(f"成功生成 {len(recommendations)} 条混合推荐")
            return recommendations

        except Exception as e:
            logging.error(f"生成混合推荐时发生错误: {str(e)}")
            raise

    @error_handler
    def optimize_hybrid_weights(self, test_data, methods=None, n_fold=5, n_trials=20):
        """
        优化混合推荐的权重

        参数:
            test_data (pandas.DataFrame): 测试数据，包含user_id, book_id, rating列
            methods (list): 要使用的推荐方法列表，默认为所有方法
            n_fold (int): 交叉验证折数
            n_trials (int): 随机搜索尝试次数

        返回:
            dict: 最优权重和对应的RMSE
        """
        logging.info("优化混合推荐的权重...")

        try:
            # 默认使用所有方法
            if methods is None:
                methods = ['user_based', 'item_based', 'matrix_factorization']

            # 检查指定的方法是否有效
            valid_methods = ['user_based', 'item_based', 'matrix_factorization']
            for method in methods:
                if method not in valid_methods:
                    raise ValueError(f"无效的推荐方法: {method}。有效值为: {valid_methods}")

            # 分割测试数据为n_fold份
            test_data_shuffled = test_data.sample(frac=1, random_state=42)  # 随机打乱
            fold_size = len(test_data_shuffled) // n_fold
            folds = [test_data_shuffled.iloc[i * fold_size:(i + 1) * fold_size] for i in range(n_fold)]

            best_rmse = float('inf')
            best_weights = None

            # 随机搜索最优权重
            import numpy as np
            np.random.seed(42)  # 设置随机种子确保结果可复现

            for trial in range(n_trials):
                # 生成随机权重，确保总和为1
                weights = np.random.dirichlet(np.ones(len(methods)))

                # 跨折交叉验证
                fold_rmses = []
                for test_fold in folds:
                    # 计算当前权重下的预测误差
                    actual_ratings = []
                    predicted_ratings = []

                    for _, row in test_fold.iterrows():
                        user_id = row['user_id']
                        book_id = row['book_id']
                        actual_rating = row['rating']

                        # 临时从训练数据中移除该评分
                        temp_ratings = self.ratings_df.copy()
                        temp_index = temp_ratings[(temp_ratings['user_id'] == user_id) &
                                                  (temp_ratings['book_id'] == book_id)].index
                        if not temp_index.empty:
                            self.ratings_df = temp_ratings.drop(temp_index)

                            try:
                                # 为该用户生成混合推荐
                                recs = self.get_hybrid_recommendations(
                                    user_id=user_id,
                                    n_recommendations=20,  # 确保足够找到目标图书
                                    methods=methods,
                                    weights=weights.tolist()
                                )

                                # 查找目标图书的预测评分
                                for rec_book_id, _, score in recs:
                                    if rec_book_id == book_id:
                                        actual_ratings.append(actual_rating)
                                        predicted_ratings.append(score)
                                        break
                            except Exception as rec_error:
                                logging.warning(f"为用户 {user_id} 生成推荐时出错: {str(rec_error)}")

                    # 恢复原始数据
                    self.ratings_df = self.ratings_df.copy()

                    # 计算当前折的RMSE
                    if actual_ratings and predicted_ratings:
                        fold_rmse = np.sqrt(mean_squared_error(actual_ratings, predicted_ratings))
                        fold_rmses.append(fold_rmse)

                # 计算平均RMSE
                if fold_rmses:
                    avg_rmse = sum(fold_rmses) / len(fold_rmses)

                    # 更新最优结果
                    if avg_rmse < best_rmse:
                        best_rmse = avg_rmse
                        best_weights = weights

                    logging.info(f"试验 {trial + 1}/{n_trials}, 权重: {weights}, RMSE: {avg_rmse:.4f}")
                else:
                    logging.warning(f"试验 {trial + 1}/{n_trials} 未产生有效评估结果")

            # 返回最优结果
            if best_weights is not None:
                result = {
                    'methods': methods,
                    'weights': best_weights.tolist(),
                    'rmse': best_rmse
                }
                logging.info(f"最优权重: {best_weights}, RMSE: {best_rmse:.4f}")
                return result
            else:
                logging.warning("未找到有效的最优权重")
                return {
                    'methods': methods,
                    'weights': [1.0 / len(methods)] * len(methods),
                    'rmse': None
                }

        except Exception as e:
            logging.error(f"优化混合推荐权重时发生错误: {str(e)}")
            raise

    @error_handler
    def evaluate_hybrid_recommendations(self, test_data, methods=None, weights=None):
        """
        评估混合推荐系统性能

        参数:
            test_data (pandas.DataFrame): 测试数据，包含user_id, book_id, rating列
            methods (list): 要使用的推荐方法列表，默认使用所有三种方法
            weights (list): 各方法的权重，默认平均权重

        返回:
            float: 均方根误差(RMSE)
        """
        logging.info("评估混合推荐方法的性能...")

        try:
            # 默认使用所有方法
            if methods is None:
                methods = ['user_based', 'item_based', 'matrix_factorization']

            # 如果未指定权重，则使用平均权重
            if weights is None:
                weights = [1.0 / len(methods)] * len(methods)

            # 检查权重和方法列表长度是否匹配
            if len(weights) != len(methods):
                raise ValueError(f"权重列表长度 ({len(weights)}) 与方法列表长度 ({len(methods)}) 不匹配")

            # 准备评估
            actual_ratings = []
            predicted_ratings = []

            # 保存原始评分数据
            original_ratings = self.ratings_df.copy()

            # 为各个方法预先准备模型
            if 'matrix_factorization' in methods:
                # 创建不包含测试数据的训练集
                train_ratings = self.ratings_df.copy()
                for _, row in test_data.iterrows():
                    user_id = row['user_id']
                    book_id = row['book_id']
                    indices = train_ratings[(train_ratings['user_id'] == user_id) &
                                            (train_ratings['book_id'] == book_id)].index
                    if not indices.empty:
                        train_ratings = train_ratings.drop(indices)

                # 使用训练数据集
                self.ratings_df = train_ratings

                # 训练矩阵分解模型
                logging.info("为混合评估训练矩阵分解模型")
                self.train_matrix_factorization(n_epochs=20)  # 使用较少的轮次加速评估

            # 预先计算相似度矩阵
            if 'user_based' in methods:
                logging.info("为混合评估计算用户相似度")
                self.calculate_user_similarity()

            if 'item_based' in methods:
                logging.info("为混合评估计算物品相似度")
                self.calculate_book_similarity()

            # 评估每个测试样例
            for _, row in test_data.iterrows():
                user_id = row['user_id']
                book_id = row['book_id']
                actual_rating = row['rating']

                try:
                    # 尝试生成推荐
                    recs = self.get_hybrid_recommendations(
                        user_id=user_id,
                        n_recommendations=100,  # 足够大以确保包含目标图书
                        methods=methods,
                        weights=weights
                    )

                    # 查找目标图书的预测评分
                    for rec_book_id, _, score in recs:
                        if rec_book_id == book_id:
                            # 确保评分在有效范围内
                            score = max(1.0, min(5.0, score))

                            actual_ratings.append(actual_rating)
                            predicted_ratings.append(score)
                            break

                except Exception as e:
                    logging.warning(f"为用户 {user_id} 评估推荐时出错: {str(e)}")

            # 恢复原始数据
            self.ratings_df = original_ratings

            # 计算RMSE
            if len(actual_ratings) > 0:
                rmse = np.sqrt(mean_squared_error(actual_ratings, predicted_ratings))
                logging.info(f"混合推荐方法的RMSE为: {rmse:.4f} (基于 {len(actual_ratings)} 条预测)")
                return rmse
            else:
                logging.warning("没有足够的数据进行混合推荐评估")
                return None

        except Exception as e:
            logging.error(f"评估混合推荐系统时发生错误: {str(e)}")
            # 确保恢复原始数据
            if 'original_ratings' in locals():
                self.ratings_df = original_ratings
            raise


@error_handler
def create_demo_data():
    """
    创建演示用的示例数据
    """
    logging.info("创建演示数据...")

    try:
        # 创建演示数据目录
        if not os.path.exists('demo_data'):
            os.makedirs('demo_data')

        # 创建用户数据
        users_data = {
            'user_id': range(1, 21),
            'username': [f'user_{i}' for i in range(1, 21)],
            'age': np.random.randint(18, 70, 20)
        }
        users_df = pd.DataFrame(users_data)
        users_df.to_csv('demo_data/users.csv', index=False, encoding='utf-8')

        # 创建图书数据
        books_data = {
            'book_id': range(1, 51),
            'title': [f'Book Title {i}' for i in range(1, 51)],
            'author': [f'Author {i % 10 + 1}' for i in range(1, 51)],
            'genre': np.random.choice(['Fiction', 'Non-fiction', 'Science', 'History', 'Philosophy'], 50)
        }
        books_df = pd.DataFrame(books_data)
        books_df.to_csv('demo_data/books.csv', index=False, encoding='utf-8')

        # 创建评分数据（稀疏矩阵）
        ratings = []
        np.random.seed(42)  # 固定随机种子，保证结果可重现

        # 每个用户评价10-20本书
        for user_id in range(1, 21):
            n_ratings = np.random.randint(10, 21)
            book_ids = np.random.choice(range(1, 51), n_ratings, replace=False)

            for book_id in book_ids:
                rating = np.random.randint(1, 6)  # 1-5的评分
                ratings.append({
                    'user_id': user_id,
                    'book_id': book_id,
                    'rating': rating,
                    'timestamp': pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(1, 365))
                })

        ratings_df = pd.DataFrame(ratings)
        ratings_df.to_csv('demo_data/ratings.csv', index=False, encoding='utf-8')

        logging.info(f"创建了 {len(users_df)} 个用户, {len(books_df)} 本书, {len(ratings_df)} 条评分记录")

    except Exception as e:
        logging.error(f"创建演示数据时发生错误: {str(e)}")
        raise


@error_handler
def get_hybrid_recommendations(self, user_id, n_recommendations=10, methods=None, weights=None):
    """
    混合推荐方法，结合多种推荐算法的结果

    参数:
        user_id (int): 用户ID
        n_recommendations (int): 推荐数量
        methods (list): 要使用的推荐方法列表，可选值: 'user_based', 'item_based', 'matrix_factorization'
                        默认使用所有三种方法
        weights (list): 各方法的权重，需与methods列表长度相同
                        默认平均权重

    返回:
        list: 推荐图书及预测评分
    """
    logging.info(f"为用户 {user_id} 生成混合推荐...")

    try:
        # 默认使用所有方法
        if methods is None:
            methods = ['user_based', 'item_based', 'matrix_factorization']

        # 检查指定的方法是否有效
        valid_methods = ['user_based', 'item_based', 'matrix_factorization']
        for method in methods:
            if method not in valid_methods:
                raise ValueError(f"无效的推荐方法: {method}。有效值为: {valid_methods}")

        # 检查用户是否存在
        if user_id not in self.users_df['user_id'].values:
            raise ValueError(f"用户ID {user_id} 不存在")

        # 如果未指定权重，则使用平均权重
        if weights is None:
            weights = [1.0 / len(methods)] * len(methods)

        # 检查权重和方法列表长度是否匹配
        if len(weights) != len(methods):
            raise ValueError(f"权重列表长度 ({len(weights)}) 与方法列表长度 ({len(methods)}) 不匹配")

        # 检查权重是否为正数且总和为1
        if any(w < 0 for w in weights) or abs(sum(weights) - 1.0) > 1e-10:
            logging.warning("权重应为正数且总和为1.0，已自动归一化")
            weights = [max(0, w) for w in weights]  # 确保权重为正
            total = sum(weights)
            if total > 0:
                weights = [w / total for w in weights]  # 归一化
            else:
                weights = [1.0 / len(methods)] * len(methods)  # 回退到平均权重

        # 生成各方法的推荐结果
        all_recommendations = {}

        # 获取用户-物品矩阵
        user_item_matrix = self.create_user_item_matrix()

        # 获取用户已评分的图书
        user_ratings = user_item_matrix.loc[user_id]
        rated_books = set(user_ratings[user_ratings > 0].index.tolist())

        # 为每种方法获取推荐
        for i, method in enumerate(methods):
            if method == 'user_based':
                # 确保已计算用户相似度
                if self.user_similarity_matrix is None:
                    self.calculate_user_similarity()

                recs = self.get_user_based_recommendations(
                    user_id=user_id,
                    n_recommendations=n_recommendations * 2  # 获取更多推荐以便后续筛选
                )

            elif method == 'item_based':
                # 确保已计算图书相似度
                if self.book_similarity_matrix is None:
                    self.calculate_book_similarity()

                recs = self.get_item_based_recommendations(
                    user_id=user_id,
                    n_recommendations=n_recommendations * 2
                )

            elif method == 'matrix_factorization':
                # 确保矩阵分解模型已训练
                if not hasattr(self, 'user_factors') or self.user_factors is None:
                    logging.info("矩阵分解模型尚未训练，正在使用默认参数训练...")
                    self.train_matrix_factorization()

                recs = self.get_matrix_factorization_recommendations(
                    user_id=user_id,
                    n_recommendations=n_recommendations * 2
                )

            # 添加到总推荐字典，标准化为(book_id, score)格式
            weight = weights[i]
            for book_id, title, score in recs:
                if book_id in all_recommendations:
                    all_recommendations[book_id] = (all_recommendations[book_id][0],
                                                    all_recommendations[book_id][1] + score * weight)
                else:
                    all_recommendations[book_id] = (title, score * weight)

        # 如果没有任何推荐结果
        if not all_recommendations:
            logging.warning(f"没有生成任何推荐结果，用户ID: {user_id}")
            return []

        # 按照加权得分排序并返回前n_recommendations个推荐
        sorted_recs = sorted(
            [(book_id, title, score) for book_id, (title, score) in all_recommendations.items()],
            key=lambda x: x[2],
            reverse=True
        )

        recommendations = sorted_recs[:n_recommendations]

        logging.info(f"成功生成 {len(recommendations)} 条混合推荐")
        return recommendations

    except Exception as e:
        logging.error(f"生成混合推荐时发生错误: {str(e)}")
        raise


@error_handler
def optimize_hybrid_weights(self, test_data, methods=None, n_fold=5, n_trials=20):
    """
    优化混合推荐的权重

    参数:
        test_data (pandas.DataFrame): 测试数据，包含user_id, book_id, rating列
        methods (list): 要使用的推荐方法列表，默认为所有方法
        n_fold (int): 交叉验证折数
        n_trials (int): 随机搜索尝试次数

    返回:
        dict: 最优权重和对应的RMSE
    """
    logging.info("优化混合推荐的权重...")

    try:
        # 默认使用所有方法
        if methods is None:
            methods = ['user_based', 'item_based', 'matrix_factorization']

        # 检查指定的方法是否有效
        valid_methods = ['user_based', 'item_based', 'matrix_factorization']
        for method in methods:
            if method not in valid_methods:
                raise ValueError(f"无效的推荐方法: {method}。有效值为: {valid_methods}")

        # 分割测试数据为n_fold份
        test_data_shuffled = test_data.sample(frac=1, random_state=42)  # 随机打乱
        fold_size = len(test_data_shuffled) // n_fold
        folds = [test_data_shuffled.iloc[i * fold_size:(i + 1) * fold_size] for i in range(n_fold)]

        best_rmse = float('inf')
        best_weights = None

        # 随机搜索最优权重
        import numpy as np
        np.random.seed(42)  # 设置随机种子确保结果可复现

        for trial in range(n_trials):
            # 生成随机权重，确保总和为1
            weights = np.random.dirichlet(np.ones(len(methods)))

            # 跨折交叉验证
            fold_rmses = []
            for test_fold in folds:
                # 计算当前权重下的预测误差
                actual_ratings = []
                predicted_ratings = []

                for _, row in test_fold.iterrows():
                    user_id = row['user_id']
                    book_id = row['book_id']
                    actual_rating = row['rating']

                    # 临时从训练数据中移除该评分
                    temp_ratings = self.ratings_df.copy()
                    temp_index = temp_ratings[(temp_ratings['user_id'] == user_id) &
                                              (temp_ratings['book_id'] == book_id)].index
                    if not temp_index.empty:
                        self.ratings_df = temp_ratings.drop(temp_index)

                        try:
                            # 为该用户生成混合推荐
                            recs = self.get_hybrid_recommendations(
                                user_id=user_id,
                                n_recommendations=20,  # 确保足够找到目标图书
                                methods=methods,
                                weights=weights.tolist()
                            )

                            # 查找目标图书的预测评分
                            for rec_book_id, _, score in recs:
                                if rec_book_id == book_id:
                                    actual_ratings.append(actual_rating)
                                    predicted_ratings.append(score)
                                    break
                        except Exception as rec_error:
                            logging.warning(f"为用户 {user_id} 生成推荐时出错: {str(rec_error)}")

                # 恢复原始数据
                self.ratings_df = self.ratings_df.copy()

                # 计算当前折的RMSE
                if actual_ratings and predicted_ratings:
                    fold_rmse = np.sqrt(mean_squared_error(actual_ratings, predicted_ratings))
                    fold_rmses.append(fold_rmse)

            # 计算平均RMSE
            if fold_rmses:
                avg_rmse = sum(fold_rmses) / len(fold_rmses)

                # 更新最优结果
                if avg_rmse < best_rmse:
                    best_rmse = avg_rmse
                    best_weights = weights

                logging.info(f"试验 {trial + 1}/{n_trials}, 权重: {weights}, RMSE: {avg_rmse:.4f}")
            else:
                logging.warning(f"试验 {trial + 1}/{n_trials} 未产生有效评估结果")

        # 返回最优结果
        if best_weights is not None:
            result = {
                'methods': methods,
                'weights': best_weights.tolist(),
                'rmse': best_rmse
            }
            logging.info(f"最优权重: {best_weights}, RMSE: {best_rmse:.4f}")
            return result
        else:
            logging.warning("未找到有效的最优权重")
            return {
                'methods': methods,
                'weights': [1.0 / len(methods)] * len(methods),
                'rmse': None
            }

    except Exception as e:
        logging.error(f"优化混合推荐权重时发生错误: {str(e)}")
        raise


@error_handler
def evaluate_hybrid_recommendations(self, test_data, methods=None, weights=None):
    """
    评估混合推荐系统性能

    参数:
        test_data (pandas.DataFrame): 测试数据，包含user_id, book_id, rating列
        methods (list): 要使用的推荐方法列表，默认使用所有三种方法
        weights (list): 各方法的权重，默认平均权重

    返回:
        float: 均方根误差(RMSE)
    """
    logging.info("评估混合推荐方法的性能...")

    try:
        # 默认使用所有方法
        if methods is None:
            methods = ['user_based', 'item_based', 'matrix_factorization']

        # 如果未指定权重，则使用平均权重
        if weights is None:
            weights = [1.0 / len(methods)] * len(methods)

        # 检查权重和方法列表长度是否匹配
        if len(weights) != len(methods):
            raise ValueError(f"权重列表长度 ({len(weights)}) 与方法列表长度 ({len(methods)}) 不匹配")

        # 准备评估
        actual_ratings = []
        predicted_ratings = []

        # 保存原始评分数据
        original_ratings = self.ratings_df.copy()

        # 为各个方法预先准备模型
        if 'matrix_factorization' in methods:
            # 创建不包含测试数据的训练集
            train_ratings = self.ratings_df.copy()
            for _, row in test_data.iterrows():
                user_id = row['user_id']
                book_id = row['book_id']
                indices = train_ratings[(train_ratings['user_id'] == user_id) &
                                        (train_ratings['book_id'] == book_id)].index
                if not indices.empty:
                    train_ratings = train_ratings.drop(indices)

            # 使用训练数据集
            self.ratings_df = train_ratings

            # 训练矩阵分解模型
            logging.info("为混合评估训练矩阵分解模型")
            self.train_matrix_factorization(n_epochs=20)  # 使用较少的轮次加速评估

        # 预先计算相似度矩阵
        if 'user_based' in methods:
            logging.info("为混合评估计算用户相似度")
            self.calculate_user_similarity()

        if 'item_based' in methods:
            logging.info("为混合评估计算物品相似度")
            self.calculate_book_similarity()

        # 评估每个测试样例
        for _, row in test_data.iterrows():
            user_id = row['user_id']
            book_id = row['book_id']
            actual_rating = row['rating']

            try:
                # 尝试生成推荐
                recs = self.get_hybrid_recommendations(
                    user_id=user_id,
                    n_recommendations=100,  # 足够大以确保包含目标图书
                    methods=methods,
                    weights=weights
                )

                # 查找目标图书的预测评分
                for rec_book_id, _, score in recs:
                    if rec_book_id == book_id:
                        # 确保评分在有效范围内
                        score = max(1.0, min(5.0, score))

                        actual_ratings.append(actual_rating)
                        predicted_ratings.append(score)
                        break

            except Exception as e:
                logging.warning(f"为用户 {user_id} 评估推荐时出错: {str(e)}")

        # 恢复原始数据
        self.ratings_df = original_ratings

        # 计算RMSE
        if len(actual_ratings) > 0:
            rmse = np.sqrt(mean_squared_error(actual_ratings, predicted_ratings))
            logging.info(f"混合推荐方法的RMSE为: {rmse:.4f} (基于 {len(actual_ratings)} 条预测)")
            return rmse
        else:
            logging.warning("没有足够的数据进行混合推荐评估")
            return None

    except Exception as e:
        logging.error(f"评估混合推荐系统时发生错误: {str(e)}")
        # 确保恢复原始数据
        if 'original_ratings' in locals():
            self.ratings_df = original_ratings
        raise

def demo():
    """
    演示系统功能
    """
    try:
        # 设置环境变量以开启调试模式
        os.environ['DEBUG'] = 'True'

        # 初始化日志
        setup_logging()
        logging.info("开始演示图书馆推荐系统")

        # 创建示例数据
        create_demo_data()

        # 初始化推荐系统
        recommender = LibraryRecommender(
            user_data_path='demo_data/users.csv',
            book_data_path='demo_data/books.csv',
            ratings_data_path='demo_data/ratings.csv'
        )

        # 显示数据摘要
        summary = recommender.data_summary()
        logging.info(f"数据摘要: {summary}")

        # 计算用户相似度
        user_similarity = recommender.calculate_user_similarity()
        logging.info(f"用户相似度矩阵形状: {user_similarity.shape}")

        # 计算图书相似度
        book_similarity = recommender.calculate_book_similarity()
        logging.info(f"图书相似度矩阵形状: {book_similarity.shape}")

        # 为用户1生成基于用户的推荐
        user_recs = recommender.get_user_based_recommendations(user_id=1, n_neighbors=3, n_recommendations=5)
        logging.info(f"用户1的基于用户的推荐: {user_recs}")

        # 为用户1生成基于物品的推荐
        item_recs = recommender.get_item_based_recommendations(user_id=1, n_recommendations=5)
        logging.info(f"用户1的基于物品的推荐: {item_recs}")

        # 训练矩阵分解模型
        recommender.train_matrix_factorization(n_factors=10, n_epochs=20)

        # 为用户1生成基于矩阵分解的推荐
        mf_recs = recommender.get_matrix_factorization_recommendations(user_id=1, n_recommendations=5)
        logging.info(f"用户1的基于矩阵分解的推荐: {mf_recs}")

        # 保存推荐结果
        recommender.save_recommendations(user_recs, 'demo_data/user_based_recommendations.csv')
        recommender.save_recommendations(item_recs, 'demo_data/item_based_recommendations.csv')
        recommender.save_recommendations(mf_recs, 'demo_data/matrix_factorization_recommendations.csv')

        # 从评分数据创建测试集（取20%）
        ratings = recommender.ratings_df
        np.random.seed(42)
        test_indices = np.random.choice(ratings.index, size=int(len(ratings) * 0.2), replace=False)
        test_data = ratings.loc[test_indices]

        # 评估基于用户的推荐
        user_rmse = recommender.evaluate_recommendations(test_data, method='user_based', n_neighbors=3)
        logging.info(f"基于用户的推荐RMSE: {user_rmse}")

        # 评估基于物品的推荐
        item_rmse = recommender.evaluate_recommendations(test_data, method='item_based')
        logging.info(f"基于物品的推荐RMSE: {item_rmse}")

        # 评估基于矩阵分解的推荐
        mf_rmse = recommender.evaluate_recommendations(test_data, method='matrix_factorization', n_factors=10,
                                                       n_epochs=20)
        logging.info(f"基于矩阵分解的推荐RMSE: {mf_rmse}")

        logging.info("演示完成")

    except Exception as e:
        logging.error(f"演示过程中发生错误: {str(e)}")
        traceback.print_exc()


if __name__ == "__main__":
    demo()