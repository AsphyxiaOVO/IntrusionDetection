import logging
import pathlib
import pickle
import sys
import time
from typing import Tuple
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
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
PKL_PATH = pathlib.Path('PKL/donnutsDataProblem.pkl')
FIG_PATH = pathlib.Path('FIG')

# 配置 matplotlib
plt.rcParams.update({'font.size': 22})
FIG_PATH.mkdir(parents=True, exist_ok=True)


def plot_heatmap(XT: np.ndarray, scores: np.ndarray, title: str, file_name: str) -> None:
    """绘制并保存热图。"""
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(XT[:, 0], XT[:, 1], c=scores, cmap='viridis', marker='o')
    plt.colorbar(scatter)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title(title)
    plt.savefig(FIG_PATH / file_name, bbox_inches='tight')
    logger.info(f"Saved heatmap: {file_name}")

def generate_torus_vectors(dims: int, number: int, r_min: float, r_max: float) -> Tuple[np.ndarray, np.ndarray]:
    """生成在环面内均匀分布的向量。"""
    vectors = np.random.uniform(-1, 1, size=(number, dims))
    radii = r_min + np.random.uniform(0, 1, number) * (r_max - r_min)
    magnitudes = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / magnitudes * radii[:, np.newaxis]
    logger.debug(f"Generated {number} torus vectors with radii between {r_min} and {r_max}.")
    return vectors[:, 0], vectors[:, 1]


def create_donut_data(contamin: int = 0) -> None:
    """创建并保存合成的 "donuts" 数据。"""
    if PKL_PATH.exists():
        logger.info(f"Pickle file {PKL_PATH} already exists. Skipping data generation.")
        return

    logger.info("Building donuts data.")
    num_objects = 1000
    Xn = np.column_stack(generate_torus_vectors(2, num_objects, 1.5, 4))
    Xb = np.column_stack(np.random.multivariate_normal([0, 0], [[0.5, 0], [0, 0.5]], num_objects).T)
    Xnt = np.column_stack(generate_torus_vectors(2, num_objects, 1.5, 4))
    Xa = np.column_stack(np.random.multivariate_normal([3.0, 3.0], [[0.25, 0], [0, 0.25]], num_objects).T)
    Xab = np.vstack([Xa, Xb])

    PKL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PKL_PATH, 'wb') as f:
        pickle.dump([Xn, Xnt, Xa, Xb, Xab], f)
    logger.info(f"Donuts data saved to {PKL_PATH}.")



def compute_diff_rf(n_trees: int = 1024, sample_size_ratio: float = 0.33, alpha0: float = 0.1) -> None:
    """训练 DiFF_RF 模型，评估异常分数，并与 Isolation Forest 进行比较。"""
    logger.info("Loading data for DiFF_RF computation.")
    try:
        with open(PKL_PATH, 'rb') as f:
            Xn, Xnt, Xa, Xb, Xab = pickle.load(f)
        logger.info("Data loaded successfully.")
    except FileNotFoundError:
        logger.error(f"Pickle file {PKL_PATH} not found. Please run create_donut_data() first.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        sys.exit(1)

    sample_size = int(sample_size_ratio * len(Xn)) if sample_size_ratio < 1 else int(sample_size_ratio)
    logger.info(f"Using sample size: {sample_size}")

    # 提取坐标
    XT = np.vstack([Xnt, Xab])

    # 绘制初始数据
    logger.info("Plotting initial donuts data.")
    plt.figure(figsize=(10, 8))
    plt.scatter(Xn[:, 0], Xn[:, 1], c='blue', label='Normal', alpha=0.6, edgecolors='w', s=50)
    plt.title('Normal Data')
    plt.savefig(FIG_PATH / 'clustersDonnuts0.pdf', bbox_inches='tight')

    plt.figure(figsize=(10, 8))
    plt.scatter(Xn[:, 0], Xn[:, 1], c='blue', label='Normal', alpha=0.6, edgecolors='w', s=50)
    plt.scatter(Xa[:, 0], Xa[:, 1], c='red', label='Anomalous', alpha=0.6, edgecolors='w', s=50)
    plt.title('Normal and Anomalous Data')
    plt.savefig(FIG_PATH / 'clustersDonnuts1.pdf', bbox_inches='tight')

    plt.figure(figsize=(10, 8))
    plt.scatter(Xn[:, 0], Xn[:, 1], c='blue', label='Normal', alpha=0.6, edgecolors='w', s=50)
    plt.scatter(Xa[:, 0], Xa[:, 1], c='red', label='Anomalous', alpha=0.6, edgecolors='w', s=50)
    plt.scatter(Xb[:, 0], Xb[:, 1], c='green', label='Background', alpha=0.6, edgecolors='w', s=50)
    plt.title('Normal, Anomalous, and Background Data')
    plt.legend()
    plt.savefig(FIG_PATH / 'clustersDonnuts2.pdf', bbox_inches='tight')

    # 训练 DiFF_RF
    logger.info("Building the DiFF_RF model.")
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
    logger.info("Computing anomaly scores with DiFF_RF.")
    sc_di, sc_ff, sc_diff_rf = diff_rf.anomaly_score(XT, alpha=alpha0)
    sc_ff = (sc_ff - sc_ff.min()) / (sc_ff.max() - sc_ff.min())
    sc_di = (sc_di - sc_di.min()) / (sc_di.max() - sc_di.min())
    sc_diff_rf = (sc_diff_rf - sc_diff_rf.min()) / (sc_diff_rf.max() - sc_diff_rf.min())

    # 绘制异常分数热图
    logger.info("Plotting anomaly scores heatmaps.")
    heatmaps = [
        (sc_ff, 'DiFF_RF (visiting frequency score) Heat Map', 'HeatMap_DiFF_RF_freqScore.pdf'),
        (sc_diff_rf, 'DiFF_RF (collective anomaly score) Heat Map', 'HeatMap_DiFF_RF_collectiveScore.pdf'),
        (sc_di, 'DiFF_RF (point-wise anomaly score) Heat Map', 'HeatMap_DiFF_RF_pointWiseScore.pdf')
    ]
    for scores, title, filename in heatmaps:
        plot_heatmap(XT, scores, title, filename)

    # 训练 Isolation Forest
    logger.info("Training Isolation Forest for comparison.")
    try:
        cif = IsolationForest(
            n_estimators=n_trees,
            max_samples=sample_size,
            contamination=0.1,
            n_jobs=8,
            random_state=42
        )
        cif.fit(Xn)
        logger.info("Isolation Forest trained successfully.")
    except Exception as e:
        logger.error(f"Error fitting Isolation Forest: {e}")
        sys.exit(1)

    # 计算 Isolation Forest 分数
    logger.info("Computing anomaly scores with Isolation Forest.")
    try:
        sc_isof = cif.decision_function(XT)
        sc_isof = (sc_isof - sc_isof.min()) / (sc_isof.max() - sc_isof.min())
    except Exception as e:
        logger.error(f"Error computing Isolation Forest scores: {e}")
        sys.exit(1)

    # 绘制 Isolation Forest Heat Map
    logger.info("Plotting Isolation Forest Heat Map.")
    sc_if = -cif.decision_function(XT)
    sc_if = (sc_if - sc_if.min()) / (sc_if.max() - sc_if.min())

    plt.figure(1003)
    xn = XT[:, 0]
    yn = XT[:, 1]
    plt.scatter(xn, yn, marker='o', c=sc_if, cmap='viridis')
    plt.colorbar()
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.title('Isolation Forest Heat Map')
    plt.savefig(FIG_PATH / 'HeatMap_IF.pdf')
    plt.show()

    # 计算并绘制 ROC 曲线
    logger.info("Calculating ROC curves.")
    y_true = np.concatenate([np.zeros(len(Xnt)), np.ones(len(Xab))])
    roc_data = {
        'DiFF_RF': sc_diff_rf,
        'Isolation Forest': sc_isof
    }
    roc_auc = {}
    plt.figure(figsize=(8, 6))
    for label, score in roc_data.items():
        fpr, tpr, _ = roc_curve(y_true, score)
        roc_auc_val = auc(fpr, tpr)
        roc_auc[label] = roc_auc_val
        plt.plot(fpr, tpr, lw=2, label=f'{label} (AUC = {roc_auc_val:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.savefig(FIG_PATH / 'ROC_Curve_Comparison.pdf', bbox_inches='tight')
    logger.info("Saved ROC_Curve_Comparison.pdf.")

    # 显示所有图形
    plt.show()

    logger.info("Completed anomaly detection and evaluation.")

    # 重新训练 Isolation Forest 并计算详细 ROC
    try:
        cif.fit(Xn)
        sc_if = -cif.decision_function(XT)
        sc_if = (sc_if - sc_if.min()) / (sc_if.max() - sc_if.min())
        logger.info("Isolation Forest retrained and scores computed.")
    except Exception as e:
        logger.error(f"Error retraining Isolation Forest: {e}")
        sys.exit(1)

    # 计算详细的 ROC 和 AUC
    logger.info("Computing detailed ROC curves and AUC scores.")
    scores = {
        'Isolation Forest': sc_if,
        'DiFF_RF (point-wise)': sc_di,
        'DiFF_RF (frequency)': sc_ff,
        'DiFF_RF (collective)': sc_diff_rf
    }
    for name, score in scores.items():
        try:
            fpr, tpr, _ = roc_curve(y_true, score)
            roc_auc_val = auc(fpr, tpr)
            logger.info(f"{name} AUC = {roc_auc_val:.4f}")
        except Exception as e:
            logger.error(f"Error computing ROC for {name}: {e}")
            sys.exit(1)

def main() -> None:
    """执行测试脚本的主函数。"""
    create_donut_data(contamin=0)
    compute_diff_rf(n_trees=256, sample_size_ratio=0.25, alpha0=1.0)

if __name__ == '__main__':
    main()