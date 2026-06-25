# 数据挖掘分析软件

基于 Flask 的 Web 数据挖掘工具，集成数据预处理、统计分析、可视化、机器学习及预测功能。

## 环境要求

- Python 3.9+
- Windows / macOS / Linux
- 浏览器（Chrome / Edge 最新版）

---

## 快速开始

### 1. 创建虚拟环境（首次运行）

```bash
python -m venv venv
```

### 2. 安装依赖

Windows：
```powershell
.\venv\Scripts\pip install -r requirements.txt
```

macOS / Linux：
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### 3. 启动服务

```bash
python run.py
```

浏览器打开 `http://localhost:5000`

---

## 功能模块

| 模块 | 说明 |
|------|------|
| 数据管理 | 上传 CSV/Excel、加载内置数据集、预览、下载、删除 |
| 数据预处理 | 缺失值处理、异常值检测、标准化/归一化、类别编码，支持撤回 |
| 统计分析 | 描述性统计、相关性分析、类别/分组统计、频数分布 |
| 数据可视化 | 折线图、柱状图、散点图、直方图、箱线图、饼图、热力图、散点矩阵图 |
| 机器学习 | K-Means、朴素贝叶斯、神经网络、SVM/SVR、决策树、KNN、Ridge 回归，支持预测 |

---

## 项目结构

```

## 常见问题

**Q: 打开页面提示"未选择数据"？**
A: 前往「数据管理」页面，上传文件或点击内置数据集加载。

**Q: 端口 5000 被占用？**
A: 修改 `run.py` 中的 `port=5000` 为其他端口。
