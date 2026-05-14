# IntrusionDetection

> 中文：基于 DiFF-RF 的异常检测复现与三维扩展实验  
> English: A DiFF-RF anomaly detection reproduction project with a 3D extension

## 中文说明

### 项目简介

本项目是 `2024-2025-1`《网络安全态势感知》课程设计，围绕论文《Random Partitioning Forest for Point-Wise and Collective Anomaly Detection -- Application to Intrusion Detection》进行改进与复现。

仓库当前公开代码的重点是：

- 复现 `DiFF-RF` 的核心森林结构与异常评分机制
- 在二维合成数据上完成训练、可视化与 ROC 对比
- 将实验进一步扩展到三维数据场景
- 与 `Isolation Forest` 进行对照实验

### 项目内容

从仓库代码来看，本项目更接近“异常检测方法复现与可视化实验平台”，而不是直接面向真实网络流量部署的完整入侵检测系统。公开脚本主要使用合成的 donut / torus 风格数据来验证方法效果，并输出热力图与 ROC 曲线。

### File Layout

| 路径 | 说明 |
| --- | --- |
| `DiFF-RF-master/DiFF_RF.py` | DiFF-RF 核心实现，包括树结构、特征权重、异常评分等 |
| `DiFF-RF-master/test.py` | 二维实验脚本：生成数据、训练 DiFF-RF、对比 Isolation Forest、绘制热图与 ROC |
| `DiFF-RF-master/test3d.py` | 三维实验脚本：将相同思路扩展到 3D 数据并输出 3D 图 |
| `DiFF-RF-master/PKL/` | 缓存生成的数据集 |
| `DiFF-RF-master/FIG/` | 保存实验结果图与 PDF 输出 |
| `README.md` | 仓库说明文档 |

### 方法概览

#### 1. DiFF-RF 核心思路

项目中的 `DiFF_RF.py` 实现了一个基于随机划分树的异常检测森林，主要包含：

- 随机选择分裂特征和分裂值
- 基于经验熵为特征分配权重
- 在叶节点计算样本与节点统计量之间的相似性
- 生成三类异常评分：
  - 点异常分数（point-wise）
  - 访问频率分数（frequency）
  - 集体异常分数（collective）

#### 2. 二维实验

`test.py` 会：

- 自动生成二维 donut 风格的正常样本、异常样本和背景样本
- 训练 `DiFF-RF`
- 使用 `Isolation Forest` 作为对照方法
- 输出热力图与 `ROC/AUC` 结果

#### 3. 三维扩展

`test3d.py` 将上述流程推广到三维数据，并生成三维散点热力图，用于观察模型在更高维实验场景下的表现。

### 运行环境

推荐使用 `Python 3.9+`，安装依赖如下：

```bash
pip install numpy matplotlib scikit-learn
```

说明：

- `pickle`、`pathlib`、`logging`、`time`、`multiprocessing` 等属于标准库。
- 代码中默认使用多进程训练，`test.py` 和 `test3d.py` 里写死了 `n_jobs=8`，如果本机核心数较少，可以按需要自行调整。

### 运行方式

进入核心目录：

```bash
cd DiFF-RF-master
```

运行二维实验：

```bash
python test.py
```

运行三维实验：

```bash
python test3d.py
```

脚本行为说明：

- 如果 `PKL/` 中不存在缓存数据，脚本会自动生成合成数据
- 如果 `FIG/` 不存在，脚本会自动创建该目录
- 运行结束后，结果图会保存到 `FIG/`

### 主要输出结果

仓库当前 README 中已经引用了以下典型输出：

- `FIG/diFFRF_3D.pdf`
- `FIG/isolationForest_3D.pdf`
- `FIG/ROC_Curve_Comparison.pdf`
- `FIG/HeatMap_DiFF_RF_collectiveScore.pdf`
- `FIG/HeatMap_DiFF_RF_freqScore.pdf`
- `FIG/HeatMap_DiFF_RF_pointWiseScore.pdf`

这些结果可以帮助观察：

- DiFF-RF 对正常区域与异常区域的区分能力
- 三种异常评分之间的差异
- DiFF-RF 与 Isolation Forest 在 ROC 曲线上的比较表现

### 项目特点与局限

- 适合用于理解 DiFF-RF 的实现细节与实验流程
- 增加了三维实验版本，使可视化与对照更完整
- 代码结构已经具备继续扩展到其他合成数据实验的基础
- 当前公开版本主要使用合成数据，不等同于真实网络流量场景下的完整 IDS 部署

## English Overview

### Summary

This repository is a course-design project that reproduces and extends the paper *Random Partitioning Forest for Point-Wise and Collective Anomaly Detection -- Application to Intrusion Detection*.

The public code mainly focuses on:

- implementing the DiFF-RF core algorithm,
- running 2D anomaly-detection experiments,
- extending the workflow to 3D data,
- and comparing the results with Isolation Forest.

### Repository Structure

| Path | Purpose |
| --- | --- |
| `DiFF-RF-master/DiFF_RF.py` | Core DiFF-RF implementation |
| `DiFF-RF-master/test.py` | 2D experiment pipeline |
| `DiFF-RF-master/test3d.py` | 3D experiment pipeline |
| `DiFF-RF-master/PKL/` | Cached synthetic datasets |
| `DiFF-RF-master/FIG/` | Saved figures and PDFs |

### Quick Start

Install dependencies:

```bash
pip install numpy matplotlib scikit-learn
```

Run the 2D experiment:

```bash
cd DiFF-RF-master
python test.py
```

Run the 3D experiment:

```bash
python test3d.py
```

### Notes

- The current public scripts use synthetic donut / torus-style datasets for method reproduction and visualization.
- The repository is well suited for studying the DiFF-RF workflow, but it is not yet a complete production intrusion-detection system for real traffic data.
- Result figures are automatically written to the `FIG/` directory.
