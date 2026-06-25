import pandas as pd
import numpy as np
import os
from faker import Faker

fake = Faker('zh_CN')
np.random.seed(42)

dataset_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'datasets')
os.makedirs(dataset_dir, exist_ok=True)


def gen_waimai():
    n = 200
    data = {
        '订单ID': [f'WM{i:04d}' for i in range(1, n+1)],
        '顾客ID': [f'S{np.random.randint(2020001, 2020501)}' for _ in range(n)],
        '商家名称': [np.random.choice(['好再来食堂', '川味轩', '兰州拉面', '沙县小吃', '黄焖鸡米饭', '麻辣香锅', '过桥米线', '周黑鸭', '肯德基', '麦当劳']) for _ in range(n)],
        '菜品类别': [np.random.choice(['中式快餐', '西式快餐', '粉面', '麻辣烫', '炸鸡', '奶茶饮品', '日韩料理', '烧烤']) for _ in range(n)],
        '价格': np.round(np.random.uniform(8, 35, n), 2),
        '评分': np.round(np.random.choice([1,2,3,4,5], n, p=[0.02,0.03,0.1,0.35,0.5]), 1),
        '配送费': np.round(np.random.choice([0, 1, 2, 3, 4, 5], n, p=[0.3,0.2,0.2,0.15,0.1,0.05]), 0),
        '配送时间(分钟)': np.random.randint(15, 60, n),
        '是否为高峰': np.random.choice(['是', '否'], n, p=[0.6, 0.4]),
        '是否准时': np.random.choice(['是', '否'], n, p=[0.85, 0.15]),
        '下单时段': np.random.choice(['早餐(6-9)', '午餐(11-14)', '下午茶(14-17)', '晚餐(17-20)', '夜宵(20-24)'], n),
        '月销量': np.random.randint(50, 5000, n),
    }
    df = pd.DataFrame(data)
    df.loc[::15, '评分'] = np.nan
    df.to_csv(os.path.join(dataset_dir, '外卖数据分析.csv'), index=False, encoding='utf-8-sig')
    print(f'[OK] 外卖数据分析.csv ({n}条)')


def gen_grades():
    n = 150
    data = {
        '学号': [f'2024{np.random.randint(10000, 99999)}' for _ in range(n)],
        '姓名': [fake.name() for _ in range(n)],
        '性别': np.random.choice(['男', '女'], n, p=[0.48, 0.52]),
        '专业': np.random.choice(['计算机科学', '软件工程', '数据科学', '人工智能', '信息管理', '电子商务'], n),
        '高等数学': np.round(np.random.normal(78, 12, n).clip(0, 100), 1),
        '线性代数': np.round(np.random.normal(75, 13, n).clip(0, 100), 1),
        '程序设计': np.round(np.random.normal(80, 14, n).clip(0, 100), 1),
        '数据结构': np.round(np.random.normal(76, 15, n).clip(0, 100), 1),
        '数据库原理': np.round(np.random.normal(77, 11, n).clip(0, 100), 1),
        '机器学习': np.round(np.random.normal(72, 16, n).clip(0, 100), 1),
        '英语': np.round(np.random.normal(79, 10, n).clip(0, 100), 1),
        '出勤率': np.round(np.random.uniform(0.6, 1.0, n), 2),
        '作业完成次数': np.random.randint(5, 15, n),
    }
    df = pd.DataFrame(data)
    df['总评'] = np.round(df[['高等数学','线性代数','程序设计','数据结构','数据库原理','机器学习','英语']].mean(axis=1), 1)
    df['等级'] = pd.cut(df['总评'], bins=[0, 60, 70, 85, 100], labels=['不及格', '及格', '良好', '优秀'])
    df.loc[::20, '机器学习'] = np.nan
    df.to_csv(os.path.join(dataset_dir, '学生成绩分析.csv'), index=False, encoding='utf-8-sig')
    print(f'[OK] 学生成绩分析.csv ({n}条)')


def gen_consumption():
    n = 300
    data = {
        '学生ID': [f'S{np.random.randint(2022001, 2022601)}' for _ in range(n)],
        '消费日期': [fake.date_between(start_date='-3M', end_date='today') for _ in range(n)],
        '消费金额': np.round(np.random.uniform(1.5, 50, n), 2),
        '消费类别': np.random.choice(['餐饮', '超市', '水果', '奶茶', '打印', '洗浴', '水费', '电费'], n),
        '支付方式': np.random.choice(['校园卡', '微信', '支付宝'], n, p=[0.6, 0.25, 0.15]),
        '消费时段': np.random.choice(['早晨', '上午', '中午', '下午', '晚上'], n),
        '星期': np.random.choice(['周一','周二','周三','周四','周五','周六','周日'], n),
        '是否在校': np.random.choice(['是', '否'], n, p=[0.9, 0.1]),
    }
    df = pd.DataFrame(data)
    df.loc[::25, '消费金额'] = np.nan
    df.to_csv(os.path.join(dataset_dir, '学生消费行为.csv'), index=False, encoding='utf-8-sig')
    print(f'[OK] 学生消费行为.csv ({n}条)')


def gen_course_eval():
    n = 120
    teachers = ['张教授', '李教授', '王教授', '赵教授', '陈教授', '刘教授', '周教授', '吴教授']
    courses = ['高等数学', '线性代数', '概率论', 'Python程序设计', '数据结构', '操作系统', '计算机网络', '软件工程']
    data = {
        '课程ID': [f'C{i:03d}' for i in range(1, n+1)],
        '课程名称': [np.random.choice(courses) for _ in range(n)],
        '授课教师': [np.random.choice(teachers) for _ in range(n)],
        '教学内容评分': np.round(np.random.uniform(3, 10, n), 1),
        '教学态度评分': np.round(np.random.uniform(3, 10, n), 1),
        '互动性评分': np.round(np.random.uniform(2, 10, n), 1),
        '作业合理性评分': np.round(np.random.uniform(3, 10, n), 1),
        '考试难度评分': np.round(np.random.uniform(1, 10, n), 1),
        '推荐指数': np.round(np.random.uniform(2, 10, n), 1),
        '选课人数': np.random.randint(30, 180, n),
        '学期': np.random.choice(['2023-2024-1', '2023-2024-2', '2024-2025-1'], n),
    }
    df = pd.DataFrame(data)
    df['综合评分'] = np.round(df[['教学内容评分','教学态度评分','互动性评分']].mean(axis=1), 1)
    df.to_csv(os.path.join(dataset_dir, '课程评价数据.csv'), index=False, encoding='utf-8-sig')
    print(f'[OK] 课程评价数据.csv ({n}条)')


def gen_campus_card():
    n = 500
    merchants = ['第一食堂', '第二食堂', '清真食堂', '教工餐厅', '咖啡厅', '超市A', '超市B',
                 '水果店', '打印店', '理发店', '体育馆', '图书馆', '医务室', '浴室']
    data = {
        '交易ID': [f'CC{i:04d}' for i in range(1, n+1)],
        '卡号': [f'Card{np.random.randint(100000, 999999)}' for _ in range(n)],
        '交易时间': [fake.date_time_between(start_date='-1M', end_date='now').strftime('%Y-%m-%d %H:%M:%S') for _ in range(n)],
        '商户名称': [np.random.choice(merchants) for _ in range(n)],
        '商户类别': [np.random.choice(['餐饮', '零售', '服务', '文体']) for _ in range(n)],
        '交易金额': np.round(np.random.uniform(0.5, 80, n), 2),
        '卡余额': np.round(np.random.uniform(0, 500, n), 2),
        '交易类型': np.random.choice(['消费', '充值', '退款'], n, p=[0.85, 0.1, 0.05]),
        '是否周末': np.random.choice(['是', '否'], n, p=[0.3, 0.7]),
    }
    df = pd.DataFrame(data)
    df.loc[::30, '交易金额'] = np.nan
    df.to_csv(os.path.join(dataset_dir, '校园卡消费.csv'), index=False, encoding='utf-8-sig')
    print(f'[OK] 校园卡消费.csv ({n}条)')


if __name__ == '__main__':
    print('正在生成数据集...')
    gen_waimai()
    gen_grades()
    gen_consumption()
    gen_course_eval()
    gen_campus_card()
    print('\n所有数据集生成完毕！')
