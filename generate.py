#!/usr/bin/env python
# -*- coding: utf-8 -*-



import numpy as np
import pandas as pd
import random
import os
import json
import logging
import time
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from faker import Faker
from collections import defaultdict, Counter


# 配置日志
def setup_logging(log_file="logs/dataset_generator.log"):
    """
    设置日志系统
    """
    # 创建日志目录
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # 配置日志格式和级别
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

    logging.info("日志系统初始化成功")


# 中文图书分类
BOOK_CATEGORIES = {
    "小说": ["悬疑推理", "科幻", "奇幻", "历史", "爱情", "武侠", "青春", "恐怖"],
    "文学": ["古典文学", "现代文学", "外国文学", "诗歌", "散文", "戏剧"],
    "人文社科": ["历史", "哲学", "社会学", "心理学", "政治", "经济", "法律", "教育"],
    "科学技术": ["数学", "物理", "化学", "生物", "天文", "地理", "医学", "计算机"],
    "艺术": ["绘画", "摄影", "音乐", "电影", "建筑", "设计"],
    "生活": ["烹饪", "旅行", "健康", "家居", "园艺", "育儿"],
    "经管": ["管理", "投资", "创业", "商业", "金融", "市场营销"]
}

# 中文出版社列表
PUBLISHERS = [
    "人民文学出版社", "商务印书馆", "中华书局", "三联书店", "北京大学出版社", "清华大学出版社",
    "上海人民出版社", "南京大学出版社", "复旦大学出版社", "浙江大学出版社", "新星出版社",
    "北京联合出版公司", "中信出版集团", "电子工业出版社", "人民邮电出版社", "机械工业出版社",
    "湖南文艺出版社", "广西师范大学出版社", "译林出版社", "人民教育出版社", "科学出版社"
]

# 美国和英国出版社列表
FOREIGN_PUBLISHERS = [
    "Penguin Random House", "HarperCollins", "Simon & Schuster", "Hachette Book Group",
    "Macmillan Publishers", "Oxford University Press", "Cambridge University Press",
    "MIT Press", "Princeton University Press", "Yale University Press", "Wiley",
    "Pearson Education", "McGraw-Hill Education", "Scholastic", "Bloomsbury Publishing"
]


class LibraryDatasetGenerator:
    """
    图书馆推荐系统数据集生成器类
    """

    def __init__(self, output_dir="generated_data", seed=None, locale='zh_CN'):
        """
        初始化数据集生成器

        参数:
            output_dir (str): 输出目录
            seed (int): 随机种子，用于重现性
            locale (str): 语言区域设置，默认为中文
        """
        # 创建输出目录
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 设置随机种子
        self.seed = seed if seed is not None else int(time.time())
        random.seed(self.seed)
        np.random.seed(self.seed)

        # 初始化Faker生成器
        self.faker = Faker(locale)
        self.faker.seed_instance(self.seed)

        # 初始化用于存储生成的数据
        self.users_df = None
        self.books_df = None
        self.ratings_df = None
        self.borrow_history_df = None

        logging.info(f"初始化数据集生成器，输出目录: {output_dir}，随机种子: {self.seed}")

    def generate_users(self, n_users=1000):
        """
        生成用户数据

        参数:
            n_users (int): 用户数量

        返回:
            pandas.DataFrame: 用户数据框
        """
        logging.info(f"开始生成 {n_users} 个用户数据...")

        # 用户数据列表
        users = []

        # 生成用户ID范围，确保没有重复
        user_ids = list(range(1, n_users + 1))

        # 性别分布
        genders = np.random.choice(['男', '女'], size=n_users, p=[0.48, 0.52])

        # 年龄分布 (加权以反映图书馆用户的真实分布)
        age_ranges = [
            (0, 14),  # 儿童
            (15, 24),  # 青少年和年轻人
            (25, 34),  # 年轻成年人
            (35, 44),  # 中年人群
            (45, 54),  # 中年人群
            (55, 64),  # 老年前期
            (65, 80)  # 老年人
        ]

        age_weights = [0.15, 0.23, 0.22, 0.18, 0.12, 0.06, 0.04]

        # 用户类型分布
        user_types = ['学生', '教师', '研究人员', '普通读者', '专业人士']
        user_type_weights = [0.45, 0.05, 0.10, 0.30, 0.10]

        # 教育程度分布
        education_levels = ['小学', '初中', '高中', '大专', '本科', '硕士', '博士']

        # 阅读偏好分布 (每个用户可能有多个偏好)
        main_categories = list(BOOK_CATEGORIES.keys())

        # 生成每个用户
        for i in range(n_users):
            # 基本信息
            user_id = user_ids[i]
            gender = genders[i]

            # 根据性别生成用户名
            if gender == '男':
                first_name = self.faker.first_name_male()
                last_name = self.faker.last_name_male()
            else:
                first_name = self.faker.first_name_female()
                last_name = self.faker.last_name_female()

            username = f"{last_name}{first_name}"

            # 生成年龄
            age_range_idx = np.random.choice(len(age_ranges), p=age_weights)
            min_age, max_age = age_ranges[age_range_idx]
            age = np.random.randint(min_age, max_age + 1)

            # 生成注册日期 (过去1-5年内)
            days_ago = np.random.randint(365, 365 * 5)
            registration_date = (datetime.now() - timedelta(days=days_ago)).strftime('%Y-%m-%d')

            # 选择用户类型
            user_type = np.random.choice(user_types, p=user_type_weights)

            # 根据年龄和用户类型确定教育程度
            if age < 15:
                education = np.random.choice(['小学', '初中'])
            elif age < 18:
                education = np.random.choice(['初中', '高中'])
            elif age < 22:
                if user_type == '学生':
                    education = np.random.choice(['高中', '大专', '本科'])
                else:
                    education = np.random.choice(['高中', '大专', '本科'], p=[0.3, 0.3, 0.4])
            else:
                if user_type == '教师' or user_type == '研究人员':
                    education = np.random.choice(['本科', '硕士', '博士'], p=[0.3, 0.4, 0.3])
                elif user_type == '专业人士':
                    education = np.random.choice(['大专', '本科', '硕士', '博士'], p=[0.2, 0.4, 0.3, 0.1])
                else:
                    education = np.random.choice(education_levels, p=[0.05, 0.1, 0.2, 0.25, 0.3, 0.08, 0.02])

            # 选择阅读偏好 (1-3个主类别)
            n_preferences = np.random.randint(1, 4)
            preferences = np.random.choice(main_categories, size=n_preferences, replace=False)

            # 为每个主类别选择0-2个子类别
            detailed_preferences = []
            for pref in preferences:
                subcategories = BOOK_CATEGORIES[pref]
                n_subcategories = np.random.randint(0, min(3, len(subcategories)) + 1)
                if n_subcategories > 0:
                    selected_subcategories = np.random.choice(subcategories, size=n_subcategories, replace=False)
                    detailed_preferences.extend([f"{pref}-{sub}" for sub in selected_subcategories])
                else:
                    detailed_preferences.append(pref)

            # 生成评分偏好 (有些用户倾向于给高分，有些倾向于给低分，大多数比较平均)
            rating_bias = np.random.choice(['high', 'low', 'neutral', 'neutral', 'neutral'],
                                           p=[0.2, 0.2, 0.2, 0.2, 0.2])
            if rating_bias == 'high':
                avg_rating = np.random.uniform(4.0, 5.0)
                rating_variance = np.random.uniform(0.2, 0.5)
            elif rating_bias == 'low':
                avg_rating = np.random.uniform(2.0, 3.5)
                rating_variance = np.random.uniform(0.2, 0.5)
            else:
                avg_rating = np.random.uniform(3.0, 4.0)
                rating_variance = np.random.uniform(0.5, 1.0)

            # 活跃度 (借阅和评分频率)
            if user_type in ['学生', '研究人员', '教师']:
                activity_level = np.random.choice(['低', '中', '高', '很高'], p=[0.1, 0.3, 0.4, 0.2])
            else:
                activity_level = np.random.choice(['低', '中', '高', '很高'], p=[0.2, 0.4, 0.3, 0.1])

            # 创建用户字典
            user = {
                'user_id': user_id,
                'username': username,
                'gender': gender,
                'age': age,
                'user_type': user_type,
                'education': education,
                'registration_date': registration_date,
                'preferences': ','.join(preferences),
                'detailed_preferences': ','.join(detailed_preferences),
                'avg_rating': avg_rating,
                'rating_variance': rating_variance,
                'activity_level': activity_level,
                'email': self.faker.email(),
                'phone': self.faker.phone_number(),
                'address': self.faker.address().replace('\n', ', ')
            }

            users.append(user)

        # 创建DataFrame
        self.users_df = pd.DataFrame(users)

        logging.info(f"成功生成 {len(self.users_df)} 条用户数据")

        return self.users_df

    def generate_books(self, n_books=5000):
        """
        生成图书数据

        参数:
            n_books (int): 图书数量

        返回:
            pandas.DataFrame: 图书数据框
        """
        logging.info(f"开始生成 {n_books} 本图书数据...")

        # 图书数据列表
        books = []

        # 生成图书ID范围
        book_ids = list(range(1, n_books + 1))

        # 主分类的分布权重 (反映图书馆常见的分类分布)
        main_categories = list(BOOK_CATEGORIES.keys())
        category_weights = [0.30, 0.25, 0.15, 0.15, 0.05, 0.05, 0.05]  # 对应main_categories中的顺序

        # 出版年份分布 (加权以反映图书馆藏书的实际分布)
        current_year = datetime.now().year
        year_ranges = [
            (current_year - 100, current_year - 50),  # 老书籍
            (current_year - 50, current_year - 20),  # 中年书籍
            (current_year - 20, current_year - 5),  # 近期书籍
            (current_year - 5, current_year)  # 新书
        ]
        year_weights = [0.05, 0.15, 0.50, 0.30]

        # 出版社分布 (中文和外文)
        all_publishers = PUBLISHERS + FOREIGN_PUBLISHERS

        # 语言分布
        languages = ['中文', '英文', '日文', '法文', '德文', '俄文', '西班牙文', '韩文']
        language_weights = [0.75, 0.18, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01]

        # 生成书名模板
        book_name_templates = [
            "《{}》",
            "《{}的{}》",
            "《{}与{}》",
            "《{}：{}》",
            "《{}之{}》",
            "《{}的{}：{}》"
        ]

        # 生成每本书
        for i in range(n_books):
            # 基本信息
            book_id = book_ids[i]

            # 选择主分类和子分类
            main_category = np.random.choice(main_categories, p=category_weights)
            subcategories = BOOK_CATEGORIES[main_category]
            subcategory = np.random.choice(subcategories)

            # 国内还是国外作者 (按语言决定)
            language = np.random.choice(languages, p=language_weights)

            if language == '中文':
                # 中文书籍
                author_surname = self.faker.last_name()
                author_given_name = "".join([self.faker.first_name() for _ in range(np.random.randint(1, 3))])
                author = f"{author_surname}{author_given_name}"
                publisher = np.random.choice(PUBLISHERS)

                # 生成中文书名
                name_template = np.random.choice(book_name_templates)
                if "{}" in name_template:
                    name_parts = []
                    for _ in range(name_template.count("{}")):
                        name_parts.append(self.faker.word())
                    title = name_template.format(*name_parts)
                else:
                    title = name_template
            else:
                # 外文书籍
                author = self.faker.name()
                publisher = np.random.choice(FOREIGN_PUBLISHERS)

                # 生成外文书名
                title = f"《{self.faker.catch_phrase()}》"

            # 选择出版年份
            year_range_idx = np.random.choice(len(year_ranges), p=year_weights)
            min_year, max_year = year_ranges[year_range_idx]
            publication_year = np.random.randint(min_year, max_year + 1)

            # 生成ISBN (10位或13位)
            if np.random.choice([True, False]):
                # 10位ISBN
                isbn = f"{np.random.randint(0, 10)}-{np.random.randint(100, 1000)}-{np.random.randint(1000, 10000)}-{np.random.choice(['X', *[str(i) for i in range(10)]])}"
            else:
                # 13位ISBN
                isbn = f"978-{np.random.randint(0, 10)}-{np.random.randint(100, 1000)}-{np.random.randint(1000, 10000)}-{np.random.randint(0, 10)}"

            # 生成页数
            if main_category in ['小说', '文学']:
                pages = np.random.randint(150, 800)
            elif main_category in ['人文社科', '科学技术']:
                pages = np.random.randint(200, 1000)
            else:
                pages = np.random.randint(80, 500)

            # 生成价格 (根据页数和出版年份)
            base_price = pages * 0.03  # 基础价格，每页大约0.03元
            if publication_year >= current_year - 5:
                price_factor = np.random.uniform(1.2, 1.5)  # 新书定价更高
            elif publication_year >= current_year - 20:
                price_factor = np.random.uniform(0.9, 1.2)
            else:
                price_factor = np.random.uniform(0.6, 0.9)  # 老书定价更低

            price = round(base_price * price_factor, 2)

            # 图书受欢迎程度 (影响借阅和评分频率)
            if publication_year >= current_year - 3:
                # 新书普遍更受欢迎
                popularity = np.random.choice(['低', '中', '高', '很高'], p=[0.1, 0.3, 0.4, 0.2])
            else:
                # 老书除经典外普遍不太受欢迎
                popularity = np.random.choice(['低', '中', '高', '很高'], p=[0.4, 0.3, 0.2, 0.1])

            # 图书品质 (影响评分)
            quality = np.random.choice(['低', '中', '高', '很高'], p=[0.1, 0.3, 0.5, 0.1])

            # 库存数量
            if popularity in ['高', '很高']:
                stock = np.random.randint(3, 10)
            else:
                stock = np.random.randint(1, 5)

            # 创建图书字典
            book = {
                'book_id': book_id,
                'title': title,
                'author': author,
                'main_category': main_category,
                'subcategory': subcategory,
                'publication_year': publication_year,
                'publisher': publisher,
                'language': language,
                'isbn': isbn,
                'pages': pages,
                'price': price,
                'popularity': popularity,
                'quality': quality,
                'stock': stock,
                'description': self.faker.paragraph(nb_sentences=5)
            }

            books.append(book)

        # 创建DataFrame
        self.books_df = pd.DataFrame(books)

        logging.info(f"成功生成 {len(self.books_df)} 条图书数据")

        return self.books_df

    def generate_borrowing_history(self, n_records=None):
        """
        生成借阅历史数据

        参数:
            n_records (int): 借阅记录数量，如果为None则基于用户活跃度自动计算

        返回:
            pandas.DataFrame: 借阅历史数据框
        """
        logging.info("开始生成借阅历史数据...")

        if self.users_df is None or self.books_df is None:
            raise ValueError("请先生成用户和图书数据")

        # 借阅历史列表
        borrowing_records = []

        # 获取用户和图书ID列表
        user_ids = self.users_df['user_id'].tolist()
        book_ids = self.books_df['book_id'].tolist()

        # 获取用户活跃度和图书受欢迎程度
        user_activity = dict(zip(self.users_df['user_id'], self.users_df['activity_level']))
        book_popularity = dict(zip(self.books_df['book_id'], self.books_df['popularity']))

        # 用户偏好
        user_preferences = {}
        for _, user in self.users_df.iterrows():
            preferences = user['preferences'].split(',')
            user_preferences[user['user_id']] = preferences

        # 书籍分类
        book_categories = dict(zip(self.books_df['book_id'], self.books_df['main_category']))

        # 当前日期
        current_date = datetime.now()

        # 为每个用户生成借阅记录
        borrow_record_id = 1

        for user_id in user_ids:
            # 根据用户活跃度确定借阅记录数量
            activity = user_activity[user_id]
            if activity == '低':
                n_borrows = np.random.randint(1, 10)
            elif activity == '中':
                n_borrows = np.random.randint(10, 30)
            elif activity == '高':
                n_borrows = np.random.randint(30, 60)
            else:  # '很高'
                n_borrows = np.random.randint(60, 150)

            # 用户注册日期
            registration_date = datetime.strptime(
                self.users_df.loc[self.users_df['user_id'] == user_id, 'registration_date'].values[0],
                '%Y-%m-%d'
            )

            # 用户的偏好类别
            user_preferred_categories = user_preferences[user_id]

            # 生成借阅记录
            for _ in range(n_borrows):
                # 选择图书 (70%概率选择符合偏好的书)
                if np.random.random() < 0.7 and user_preferred_categories:
                    # 筛选符合用户偏好的书籍
                    preferred_books = [
                        book_id for book_id in book_ids
                        if book_categories[book_id] in user_preferred_categories
                    ]

                    if preferred_books:
                        book_id = np.random.choice(preferred_books)
                    else:
                        book_id = np.random.choice(book_ids)
                else:
                    book_id = np.random.choice(book_ids)

                # 确定借阅日期 (在用户注册日期和当前日期之间)
                days_since_registration = (current_date - registration_date).days
                if days_since_registration <= 0:
                    continue  # 跳过注册日期在未来的异常情况

                days_ago = np.random.randint(0, days_since_registration)
                borrow_date = (current_date - timedelta(days=days_ago)).strftime('%Y-%m-%d')

                # 确定预计归还日期 (通常为30天后)
                expected_return_date = (current_date - timedelta(days=days_ago) + timedelta(days=30)).strftime(
                    '%Y-%m-%d')

                # 确定实际归还日期
                if days_ago >= 30:
                    # 90%的概率按时归还
                    if np.random.random() < 0.9:
                        actual_days = np.random.randint(20, 35)  # 提前或稍微延迟几天
                    else:
                        actual_days = np.random.randint(35, 60)  # 延迟较多

                    actual_return_date = (
                                current_date - timedelta(days=days_ago) + timedelta(days=actual_days)).strftime(
                        '%Y-%m-%d')
                    status = '已归还'
                else:
                    # 正在借阅中
                    actual_return_date = None
                    status = '借阅中'

                # 创建借阅记录
                record = {
                    'borrow_id': borrow_record_id,
                    'user_id': user_id,
                    'book_id': book_id,
                    'borrow_date': borrow_date,
                    'expected_return_date': expected_return_date,
                    'actual_return_date': actual_return_date,
                    'status': status
                }

                borrowing_records.append(record)
                borrow_record_id += 1

        # 如果指定了记录数量，则进行截断
        if n_records is not None and n_records < len(borrowing_records):
            borrowing_records = borrowing_records[:n_records]

        # 创建DataFrame
        self.borrow_history_df = pd.DataFrame(borrowing_records)

        logging.info(f"成功生成 {len(self.borrow_history_df)} 条借阅历史数据")

        return self.borrow_history_df

    def generate_ratings(self, n_ratings=None, rating_probability=0.5):
        """
        生成评分数据

        参数:
            n_ratings (int): 评分记录数量，如果为None则基于借阅历史自动计算
            rating_probability (float): 用户为已归还书籍评分的概率

        返回:
            pandas.DataFrame: 评分数据框
        """
        logging.info("开始生成评分数据...")

        if self.users_df is None or self.books_df is None or self.borrow_history_df is None:
            raise ValueError("请先生成用户、图书和借阅历史数据")

        # 评分列表
        ratings = []

        # 获取已归还的借阅记录
        returned_borrows = self.borrow_history_df[self.borrow_history_df['status'] == '已归还']

        # 用户评分偏好
        user_avg_ratings = dict(zip(self.users_df['user_id'], self.users_df['avg_rating']))
        user_rating_variances = dict(zip(self.users_df['user_id'], self.users_df['rating_variance']))

        # 书籍质量
        book_quality = {}
        for _, book in self.books_df.iterrows():
            if book['quality'] == '低':
                quality_score = np.random.uniform(1.5, 3.0)
            elif book['quality'] == '中':
                quality_score = np.random.uniform(2.5, 4.0)
            elif book['quality'] == '高':
                quality_score = np.random.uniform(3.5, 4.5)
            else:  # '很高'
                quality_score = np.random.uniform(4.0, 5.0)

            book_quality[book['book_id']] = quality_score

        # 为每条已归还的借阅记录生成评分
        rating_id = 1

        for _, borrow in returned_borrows.iterrows():
            # 用户是否对这本书进行评分
            if np.random.random() < rating_probability:
                user_id = borrow['user_id']
                book_id = borrow['book_id']

                # 获取用户评分偏好
                avg_rating = user_avg_ratings[user_id]
                rating_variance = user_rating_variances[user_id]

                # 获取书籍质量
                book_quality_score = book_quality[book_id]

                # 计算评分 (结合用户偏好和书籍质量)
                # 使用正态分布生成随机评分
                raw_rating = np.random.normal(
                    (avg_rating + book_quality_score) / 2,  # 平均值是用户偏好和书籍质量的均值
                    rating_variance  # 方差来自用户的评分变异性
                )

                # 将评分限制在1-5范围内，并四舍五入到最接近的0.5
                rating = max(1, min(5, round(raw_rating * 2) / 2))

                # 确定评分日期 (通常在归还日期后的一周内)
                return_date = datetime.strptime(borrow['actual_return_date'], '%Y-%m-%d')
                days_after_return = np.random.randint(0, 7)
                rating_date = (return_date + timedelta(days=days_after_return)).strftime('%Y-%m-%d')

                # 检查评分日期是否在未来
                if datetime.strptime(rating_date, '%Y-%m-%d') > datetime.now():
                    rating_date = datetime.now().strftime('%Y-%m-%d')

                # 根据评分生成评论
                comment = None
                comment_probability = 0.3  # 30%的评分有评论
                if np.random.random() < comment_probability:
                    if rating >= 4.5:
                        comments = [
                            "非常棒的书，强烈推荐！",
                            "这是我读过的最好的书之一。",
                            "内容超出预期，值得一读。",
                            "绝对的经典，每一页都很精彩。",
                            "这本书改变了我的看法，太棒了。"
                        ]
                    elif rating >= 3.5:
                        comments = [
                            "很好的一本书，内容充实。",
                            "整体来说不错，有些章节特别精彩。",
                            "推荐阅读，内容有深度。",
                            "观点很有趣，写作风格也不错。",
                            "这本书给了我很多启发，值得一读。"
                        ]
                    elif rating >= 2.5:
                        comments = [
                            "还可以，但有些部分比较枯燥。",
                            "内容一般，不过有几个亮点。",
                            "有些见解很独特，但整体平平。",
                            "可以一读，但不要期望太高。",
                            "部分内容有价值，但结构不太紧凑。"
                        ]
                    else:
                        comments = [
                            "内容比较薄弱，不太推荐。",
                            "失望，与描述的不符。",
                            "写作风格不佳，难以继续阅读。",
                            "这本书可以跳过，没什么特别之处。",
                            "内容过于基础，不适合有经验的读者。"
                        ]

                    comment = np.random.choice(comments)

                # 创建评分记录
                record = {
                    'rating_id': rating_id,
                    'user_id': user_id,
                    'book_id': book_id,
                    'borrow_id': borrow['borrow_id'],
                    'rating': rating,
                    'rating_date': rating_date,
                    'comment': comment
                }

                ratings.append(record)
                rating_id += 1

        # 如果指定了评分数量，则进行截断
        if n_ratings is not None and n_ratings < len(ratings):
            ratings = ratings[:n_ratings]

        # 创建DataFrame
        self.ratings_df = pd.DataFrame(ratings)

        logging.info(f"成功生成 {len(self.ratings_df)} 条评分数据")

        return self.ratings_df

    def analyze_dataset(self):
        """
        分析生成的数据集

        返回:
            dict: 数据集分析结果
        """
        logging.info("开始分析数据集...")

        analysis = {}

        if self.users_df is not None:
            analysis['users'] = {
                'count': len(self.users_df),
                'gender_distribution': self.users_df['gender'].value_counts().to_dict(),
                'age_distribution': {
                    '0-14': len(self.users_df[self.users_df['age'] <= 14]),
                    '15-24': len(self.users_df[(self.users_df['age'] > 14) & (self.users_df['age'] <= 24)]),
                    '25-34': len(self.users_df[(self.users_df['age'] > 24) & (self.users_df['age'] <= 34)]),
                    '35-44': len(self.users_df[(self.users_df['age'] > 34) & (self.users_df['age'] <= 44)]),
                    '45-54': len(self.users_df[(self.users_df['age'] > 44) & (self.users_df['age'] <= 54)]),
                    '55-64': len(self.users_df[(self.users_df['age'] > 54) & (self.users_df['age'] <= 64)]),
                    '65+': len(self.users_df[self.users_df['age'] > 64])
                },
                'user_type_distribution': self.users_df['user_type'].value_counts().to_dict(),
                'education_distribution': self.users_df['education'].value_counts().to_dict(),
                'activity_level_distribution': self.users_df['activity_level'].value_counts().to_dict()
            }

            # 分析用户偏好
            all_preferences = []
            for prefs in self.users_df['preferences'].str.split(','):
                all_preferences.extend(prefs)

            preference_counts = Counter(all_preferences)
            analysis['users']['preference_distribution'] = {k: v for k, v in preference_counts.most_common()}

        if self.books_df is not None:
            analysis['books'] = {
                'count': len(self.books_df),
                'category_distribution': self.books_df['main_category'].value_counts().to_dict(),
                'subcategory_distribution': self.books_df['subcategory'].value_counts().to_dict(),
                'language_distribution': self.books_df['language'].value_counts().to_dict(),
                'publication_year_distribution': {
                    '先前': len(self.books_df[self.books_df['publication_year'] <= datetime.now().year - 50]),
                    '过去50年': len(self.books_df[(self.books_df['publication_year'] > datetime.now().year - 50) &
                                                  (self.books_df['publication_year'] <= datetime.now().year - 20)]),
                    '过去20年': len(self.books_df[(self.books_df['publication_year'] > datetime.now().year - 20) &
                                                  (self.books_df['publication_year'] <= datetime.now().year - 5)]),
                    '近5年': len(self.books_df[self.books_df['publication_year'] > datetime.now().year - 5])
                },
                'popularity_distribution': self.books_df['popularity'].value_counts().to_dict(),
                'quality_distribution': self.books_df['quality'].value_counts().to_dict(),
                'average_price': self.books_df['price'].mean(),
                'average_pages': self.books_df['pages'].mean()
            }

        if self.borrow_history_df is not None:
            analysis['borrowing'] = {
                'count': len(self.borrow_history_df),
                'status_distribution': self.borrow_history_df['status'].value_counts().to_dict(),
                'average_borrowing_per_user': len(self.borrow_history_df) / len(
                    self.users_df) if self.users_df is not None else None,
                'borrowing_distribution': self.borrow_history_df['user_id'].value_counts().describe().to_dict()
            }

            # 计算各类别的借阅比例
            if self.books_df is not None:
                book_categories = dict(zip(self.books_df['book_id'], self.books_df['main_category']))
                borrowed_categories = [book_categories.get(book_id) for book_id in self.borrow_history_df['book_id']]
                category_counts = Counter(borrowed_categories)
                analysis['borrowing']['category_distribution'] = {k: v for k, v in category_counts.most_common()}

        if self.ratings_df is not None:
            analysis['ratings'] = {
                'count': len(self.ratings_df),
                'rating_distribution': self.ratings_df['rating'].value_counts().sort_index().to_dict(),
                'average_rating': self.ratings_df['rating'].mean(),
                'comments_percentage': (self.ratings_df['comment'].notna().sum() / len(self.ratings_df)) * 100,
                'rating_per_user': self.ratings_df['user_id'].value_counts().describe().to_dict(),
                'rating_per_book': self.ratings_df['book_id'].value_counts().describe().to_dict()
            }

            # 计算矩阵稀疏度
            if self.users_df is not None and self.books_df is not None:
                total_possible_ratings = len(self.users_df) * len(self.books_df)
                sparsity = 1.0 - (len(self.ratings_df) / total_possible_ratings)
                analysis['ratings']['sparsity'] = sparsity

        # 保存分析结果
        with open(os.path.join(self.output_dir, "dataset_analysis.json"), "w", encoding="utf-8") as f:
            json.dump(analysis, f, ensure_ascii=False, indent=4)

        logging.info("数据集分析完成")

        return analysis

    def generate_complete_dataset(self, n_users=1000, n_books=5000, rating_probability=0.5):
        """
        生成完整的数据集

        参数:
            n_users (int): 用户数量
            n_books (int): 图书数量
            rating_probability (float): 评分概率

        返回:
            tuple: (用户数据框, 图书数据框, 借阅历史数据框, 评分数据框)
        """
        logging.info(f"开始生成完整数据集，用户数: {n_users}，图书数: {n_books}")

        # 生成用户数据
        self.generate_users(n_users)

        # 生成图书数据
        self.generate_books(n_books)

        # 生成借阅历史
        self.generate_borrowing_history()

        # 生成评分数据
        self.generate_ratings(rating_probability=rating_probability)

        # 分析数据集
        self.analyze_dataset()

        # 保存数据
        self.save_dataset()

        logging.info("完整数据集生成完成")

        return self.users_df, self.books_df, self.borrow_history_df, self.ratings_df

    def save_dataset(self):
        """
        保存生成的数据集
        """
        logging.info(f"保存数据集到 {self.output_dir}")

        # 保存用户数据
        if self.users_df is not None:
            self.users_df.to_csv(os.path.join(self.output_dir, "users.csv"), index=False, encoding="utf-8")
            logging.info(f"用户数据已保存：{len(self.users_df)} 条记录")

        # 保存图书数据
        if self.books_df is not None:
            self.books_df.to_csv(os.path.join(self.output_dir, "books.csv"), index=False, encoding="utf-8")
            logging.info(f"图书数据已保存：{len(self.books_df)} 条记录")

        # 保存借阅历史
        if self.borrow_history_df is not None:
            self.borrow_history_df.to_csv(os.path.join(self.output_dir, "borrow_history.csv"), index=False,
                                          encoding="utf-8")
            logging.info(f"借阅历史已保存：{len(self.borrow_history_df)} 条记录")

        # 保存评分数据
        if self.ratings_df is not None:
            self.ratings_df.to_csv(os.path.join(self.output_dir, "ratings.csv"), index=False, encoding="utf-8")
            logging.info(f"评分数据已保存：{len(self.ratings_df)} 条记录")

        # 保存协同过滤需要的简化数据集
        if self.ratings_df is not None:
            cf_ratings = self.ratings_df[['user_id', 'book_id', 'rating']]
            cf_ratings.to_csv(os.path.join(self.output_dir, "cf_ratings.csv"), index=False, encoding="utf-8")
            logging.info(f"协同过滤评分数据已保存：{len(cf_ratings)} 条记录")

    def visualize_dataset(self):
        """
        可视化数据集特性
        """
        logging.info("开始可视化数据集...")

        # 添加以下代码，设置matplotlib支持中文
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']  # 优先使用的中文字体
        plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
        plt.rcParams['font.family'] = 'sans-serif'  # 使用无衬线字体

        # 创建可视化目录
        viz_dir = os.path.join(self.output_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)

        plt.style.use('ggplot')

        # 1. 用户年龄分布
        if self.users_df is not None:
            plt.figure(figsize=(10, 6))
            sns.histplot(self.users_df['age'], bins=20, kde=True)
            plt.title('用户年龄分布')
            plt.xlabel('年龄')
            plt.ylabel('用户数量')
            plt.savefig(os.path.join(viz_dir, 'user_age_distribution.png'))
            plt.close()

            # 用户类型分布
            plt.figure(figsize=(10, 6))
            user_types = self.users_df['user_type'].value_counts()
            user_types.plot(kind='bar')
            plt.title('用户类型分布')
            plt.xlabel('用户类型')
            plt.ylabel('数量')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'user_type_distribution.png'))
            plt.close()

            # 用户偏好分布
            plt.figure(figsize=(12, 6))
            all_preferences = []
            for prefs in self.users_df['preferences'].str.split(','):
                all_preferences.extend(prefs)

            preference_counts = pd.Series(Counter(all_preferences)).sort_values(ascending=False)
            preference_counts.plot(kind='bar')
            plt.title('用户阅读偏好分布')
            plt.xlabel('图书类别')
            plt.ylabel('喜好用户数量')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'user_preference_distribution.png'))
            plt.close()

        # 2. 图书分类分布
        if self.books_df is not None:
            plt.figure(figsize=(12, 6))
            category_counts = self.books_df['main_category'].value_counts()
            category_counts.plot(kind='bar')
            plt.title('图书主分类分布')
            plt.xlabel('主分类')
            plt.ylabel('图书数量')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'book_category_distribution.png'))
            plt.close()

            # 图书出版年份分布
            plt.figure(figsize=(12, 6))
            sns.histplot(self.books_df['publication_year'], bins=30, kde=True)
            plt.title('图书出版年份分布')
            plt.xlabel('出版年份')
            plt.ylabel('图书数量')
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'book_publication_year.png'))
            plt.close()

            # 图书语言分布
            plt.figure(figsize=(10, 6))
            language_counts = self.books_df['language'].value_counts()
            language_counts.plot(kind='pie', autopct='%1.1f%%')
            plt.title('图书语言分布')
            plt.ylabel('')
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'book_language_distribution.png'))
            plt.close()

        # 3. 评分分布
        if self.ratings_df is not None:
            plt.figure(figsize=(10, 6))
            sns.countplot(x='rating', data=self.ratings_df)
            plt.title('评分分布')
            plt.xlabel('评分')
            plt.ylabel('数量')
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'rating_distribution.png'))
            plt.close()

            # 用户评分数量分布
            plt.figure(figsize=(10, 6))
            user_rating_counts = self.ratings_df['user_id'].value_counts()
            sns.histplot(user_rating_counts, bins=30, kde=True)
            plt.title('用户评分数量分布')
            plt.xlabel('每用户评分数量')
            plt.ylabel('用户数量')
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'user_rating_count_distribution.png'))
            plt.close()

            # 图书评分数量分布
            plt.figure(figsize=(10, 6))
            book_rating_counts = self.ratings_df['book_id'].value_counts()
            sns.histplot(book_rating_counts, bins=30, kde=True)
            plt.title('图书评分数量分布')
            plt.xlabel('每本书评分数量')
            plt.ylabel('图书数量')
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'book_rating_count_distribution.png'))
            plt.close()

        # 4. 借阅历史分析
        if self.borrow_history_df is not None and self.books_df is not None:
            # 各类别借阅比例
            book_categories = dict(zip(self.books_df['book_id'], self.books_df['main_category']))
            borrowed_categories = [book_categories.get(book_id) for book_id in self.borrow_history_df['book_id']]

            plt.figure(figsize=(12, 6))
            category_counts = pd.Series(Counter(borrowed_categories)).sort_values(ascending=False)
            category_counts.plot(kind='bar')
            plt.title('各类别图书借阅比例')
            plt.xlabel('图书类别')
            plt.ylabel('借阅次数')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'category_borrowing_distribution.png'))
            plt.close()

            # 用户借阅数量分布
            plt.figure(figsize=(10, 6))
            user_borrow_counts = self.borrow_history_df['user_id'].value_counts()
            sns.histplot(user_borrow_counts, bins=30, kde=True)
            plt.title('用户借阅数量分布')
            plt.xlabel('每用户借阅数量')
            plt.ylabel('用户数量')
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'user_borrowing_count_distribution.png'))
            plt.close()

        # 5. 用户-物品矩阵稀疏性可视化
        if self.ratings_df is not None and self.users_df is not None and self.books_df is not None:
            # 随机抽样500个用户和500本书进行可视化
            n_users_sample = min(500, len(self.users_df))
            n_books_sample = min(500, len(self.books_df))

            user_sample = np.random.choice(self.users_df['user_id'].unique(), n_users_sample, replace=False)
            book_sample = np.random.choice(self.books_df['book_id'].unique(), n_books_sample, replace=False)

            ratings_sample = self.ratings_df[
                self.ratings_df['user_id'].isin(user_sample) &
                self.ratings_df['book_id'].isin(book_sample)
                ]

            # 创建用户-物品矩阵
            user_item_matrix = pd.pivot_table(
                ratings_sample,
                values='rating',
                index='user_id',
                columns='book_id',
                fill_value=0
            )

            plt.figure(figsize=(10, 8))
            plt.spy(user_item_matrix, markersize=0.5)
            plt.title('用户-物品矩阵稀疏性可视化 (样本)')
            plt.xlabel('图书ID')
            plt.ylabel('用户ID')
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'user_item_matrix_sparsity.png'))
            plt.close()

            # 计算并显示稀疏度
            sparsity = 1.0 - (len(ratings_sample) / (n_users_sample * n_books_sample))
            with open(os.path.join(viz_dir, 'sparsity_info.txt'), 'w') as f:
                f.write(f"用户-物品矩阵稀疏度: {sparsity:.4f}\n")
                f.write(f"样本大小: {n_users_sample} 用户 x {n_books_sample} 图书\n")
                f.write(f"总可能评分数: {n_users_sample * n_books_sample}\n")
                f.write(f"实际评分数: {len(ratings_sample)}\n")

        logging.info(f"数据集可视化完成，结果保存到 {viz_dir}")


def generate_demo_dataset():
    """
    生成演示数据集
    """
    # 设置日志
    setup_logging()

    # 初始化生成器
    generator = LibraryDatasetGenerator(output_dir="library_dataset")

    # 生成小型演示数据集
    users_df, books_df, borrow_history_df, ratings_df = generator.generate_complete_dataset(
        n_users=500,
        n_books=1000,
        rating_probability=0.6
    )

    # 可视化数据集
    generator.visualize_dataset()

    # 打印数据集统计信息
    print("\n=== 数据集生成完成 ===")
    print(f"生成了 {len(users_df)} 个用户")
    print(f"生成了 {len(books_df)} 本图书")
    print(f"生成了 {len(borrow_history_df)} 条借阅记录")
    print(f"生成了 {len(ratings_df)} 条评分记录")
    print(f"矩阵稀疏度: {1.0 - (len(ratings_df) / (len(users_df) * len(books_df))):.6f}")
    print("\n数据集保存在 'library_dataset' 目录下")
    print("可视化结果保存在 'library_dataset/visualizations' 目录下")


if __name__ == "__main__":
    # 运行演示
    generate_demo_dataset()