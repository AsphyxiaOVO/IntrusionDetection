import logging
import pathlib
import pickle
import sys
import time
from typing import Tuple

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import auc, roc_curve
from DiFF_RF import DiFF_TreeEnsemble

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# 配置常量和路径
PKL_PATH = pathlib.Path('PKL/donnutsDataProblem3d.pkl')
FIG_PATH = pathlib.Path('FIG')

# 配置 matplotlib
plt.rcParams.update({'font.size': 22})
FIG_PATH.mkdir(parents=True, exist_ok=True)


def plot_heatmap_3d(XT: np.ndarray, scores: np.ndarray, title: str, file_name: str) -> None:
    """绘制并保存三维热图。"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(XT[:, 0], XT[:, 1], XT[:, 2], c=scores, cmap='viridis', marker='o')
    fig.colorbar(scatter)
    ax.set_xlabel('X axis', fontsize=14)
    ax.set_ylabel('Y axis', fontsize=14)
    ax.set_zlabel('Z axis', fontsize=14)
    ax.set_title(title, fontsize=16)
    plt.savefig(FIG_PATH / file_name, bbox_inches='tight')
    logger.info(f"Saved 3D heatmap: {file_name}")


def generate_torus_vectors_3d(dims: int, number: int, r_min: float, r_max: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """生成在三维环面内均匀分布的向量。"""
    vectors = np.random.uniform(-1, 1, size=(number, dims))
    radii = r_min + np.random.uniform(0, 1, number) * (r_max - r_min)
    magnitudes = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / magnitudes * radii[:, np.newaxis]
    logger.debug(f"Generated {number} 3D torus vectors with radii between {r_min} and {r_max}.")
    return vectors[:, 0], vectors[:, 1], vectors[:, 2]


def create_donut_data_3d(contamin: int = 0) -> None:
    """创建并保存合成的三维 "donuts" 数据。"""
    if PKL_PATH.exists():
        logger.info(f"Pickle file {PKL_PATH} already exists. Skipping data generation.")
        return

    logger.info("Building 3D donuts data.")
    num_objects = 1000
    # 生成环面数据
    Xn = np.column_stack(generate_torus_vectors_3d(3, num_objects, 1.5, 4))
    Xb = np.column_stack(np.random.multivariate_normal([0, 0, 0], [[0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]], num_objects).T)
    Xnt = np.column_stack(generate_torus_vectors_3d(3, num_objects, 1.5, 4))
    Xa = np.column_stack(np.random.multivariate_normal([3.0, 3.0, 3.0], [[0.25, 0, 0], [0, 0.25, 0], [0, 0, 0.25]], num_objects).T)
    Xab = np.vstack([Xa, Xb])

    PKL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PKL_PATH, 'wb') as f:
        pickle.dump([Xn, Xnt, Xa, Xb, Xab], f)
    logger.info(f"3D donuts data saved to {PKL_PATH}.")


def compute_diff_rf_3d(n_trees: int = 1024, sample_size_ratio: float = 0.33, alpha0: float = 0.1) -> None:
    """训练 DiFF_RF 模型，评估异常分数，并与 Isolation Forest 进行比较（3D版本）。"""
    logger.info("Loading data for 3D DiFF_RF computation.")
    try:
        with open(PKL_PATH, 'rb') as f:
            Xn, Xnt, Xa, Xb, Xab = pickle.load(f)
        logger.info("Data loaded successfully.")
    except FileNotFoundError:
        logger.error(f"Pickle file {PKL_PATH} not found. Please run create_donut_data_3d() first.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        sys.exit(1)

    sample_size = int(sample_size_ratio * len(Xn)) if sample_size_ratio < 1 else int(sample_size_ratio)
    logger.info(f"Using sample size: {sample_size}")

    # 提取三维数据
    XT = np.vstack([Xnt, Xab])

    # 绘制初始数据
    logger.info("Plotting initial 3D donuts data.")
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(Xn[:, 0], Xn[:, 1], Xn[:, 2], c='blue', label='Normal', alpha=0.6, edgecolors='w', s=50)
    ax.set_title('Normal Data')
    plt.savefig(FIG_PATH / 'clustersDonnuts_3d_0.pdf', bbox_inches='tight')

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(Xn[:, 0], Xn[:, 1], Xn[:, 2], c='blue', label='Normal', alpha=0.6, edgecolors='w', s=50)
    ax.scatter(Xa[:, 0], Xa[:, 1], Xa[:, 2], c='red', label='Anomalous', alpha=0.6, edgecolors='w', s=50)
    ax.set_title('Normal and Anomalous Data')
    plt.savefig(FIG_PATH / 'clustersDonnuts_3d_1.pdf', bbox_inches='tight')

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(Xn[:, 0], Xn[:, 1], Xn[:, 2], c='blue', label='Normal', alpha=0.6, edgecolors='w', s=50)
    ax.scatter(Xa[:, 0], Xa[:, 1], Xa[:, 2], c='red', label='Anomalous', alpha=0.6, edgecolors='w', s=50)
    ax.scatter(Xb[:, 0], Xb[:, 1], Xb[:, 2], c='green', label='Background', alpha=0.6, edgecolors='w', s=50)
    ax.set_title('Normal, Anomalous, and Background Data')
    ax.legend()
    plt.savefig(FIG_PATH / 'clustersDonnuts_3d_2.pdf', bbox_inches='tight')

    # 训练 DiFF_RF
    logger.info("Building the DiFF_RF model for 3D data.")
    diff_rf = DiFF_TreeEnsemble(sample_size=sample_size, n_trees=n_trees)
    fit_start = time.time()
    try:
        diff_rf.fit(Xn, n_jobs=8)
        logger.info("DiFF_RF model fitted successfully.")
    except Exception as e:
        logger.error(f"Error fitting DiFF_RF model: {e}")
        sys.exit(1)
    logger.info(f"Fit time: {time.time() - fit_start:.2f}s")

    # 计算异常分数
    logger.info("Computing anomaly scores with DiFF_RF (3D).")
    sc_di, sc_ff, sc_diff_rf = diff_rf.anomaly_score(XT, alpha=alpha0)
    sc_ff = np.clip(sc_ff, 0, 1)

    # 绘制异常分数的热图
    plot_heatmap_3d(XT, sc_diff_rf, 'DiFF_RF Anomaly Scores', 'diFFRF_3D.pdf')

    # Isolation Forest
    logger.info("Training Isolation Forest for 3D data.")
    isolation_forest = IsolationForest(n_estimators=100, max_samples=sample_size)
    fit_start = time.time()
    isolation_forest.fit(Xn)
    logger.info(f"Isolation Forest fitted in {time.time() - fit_start:.2f}s")
    sc_if = isolation_forest.decision_function(XT)
    sc_if = np.clip(sc_if, 0, 1)

    # 绘制Isolation Forest的热图
    plot_heatmap_3d(XT, sc_if, 'Isolation Forest Anomaly Scores', 'isolationForest_3D.pdf')


def main() -> None:
    """执行3D版本的测试脚本的主函数。"""
    create_donut_data_3d(contamin=0)
    compute_diff_rf_3d(n_trees=256, sample_size_ratio=0.25, alpha0=1.0)

if __name__ == "__main__":
    main()
