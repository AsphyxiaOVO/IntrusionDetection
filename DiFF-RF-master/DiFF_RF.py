import logging
import time
from multiprocessing import Pool
from typing import Any, List, Optional, Tuple

import numpy as np
import random as rn

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def get_split(X: np.ndarray) -> float:
    """随机选择一个分割值。"""
    split_value = np.random.uniform(X.min(), X.max())
    logger.debug(f"Selected split value: {split_value} from range ({X.min()}, {X.max()})")
    return split_value


def similarity_score(S: np.ndarray, node: 'LeafNode', alpha: float) -> np.ndarray:
    """计算加权相似性分数。"""
    if S.size == 0:
        logger.debug("Empty set S provided to similarity_score.")
        return np.array([0.0])
    normalized = (S - node.M) / node.Mstd
    scores = 2 ** (-alpha * (np.sum(normalized ** 2, axis=1) / S.shape[1]))
    logger.debug(f"Computed similarity scores with alpha={alpha}: {scores}")
    return scores


def empirical_entropy(hist: np.ndarray) -> float:
    """计算经验熵。"""
    h = hist.astype(np.float64)
    total = h.sum()
    if total <= 0 or np.any(h < 0):
        logger.debug("Invalid histogram for entropy calculation.")
        return 0.0
    h /= total
    entropy = -(h * np.ma.log2(h)).filled(0).sum()
    logger.debug(f"Computed empirical entropy: {entropy}")
    return entropy


def weight_feature(s: np.ndarray, nbins: int, wmin: float = 0.02) -> float:
    """基于经验熵计算特征权重。"""
    mins, maxs = s.min(), s.max()
    if not np.isfinite(mins) or not np.isfinite(maxs) or np.abs(mins - maxs) < 1e-300:
        logger.debug("Feature with non-finite or constant values encountered.")
        return 1e-4
    ent = empirical_entropy(np.histogram(s, bins=nbins)[0]) / np.log2(nbins)
    return max(1 - ent, wmin) if np.isfinite(ent) else wmin


class BaseNode:
    """DiFF 树的基节点。"""

    def __init__(self, X: np.ndarray, height: int, sample_size: int, Xp: Optional[np.ndarray] = None) -> None:
        self.height = height + 1
        self.size = len(X)
        self.n_nodes = 1
        self.freq = self.size / sample_size
        if self.size > 0:
            self.M = np.mean(X, axis=0)
            self.Mstd = np.std(X, axis=0) if self.size > 10 else np.ones(X.shape[1])
            self.Mstd[self.Mstd == 0] = 1e-2
            logger.debug(f"Initialized BaseNode with size {self.size}.")
        else:
            self.M = np.mean(Xp, axis=0)
            self.Mstd = np.std(Xp, axis=0) if len(Xp) > 10 else np.ones(Xp.shape[1])
            self.Mstd[self.Mstd == 0] = 1e-2
            logger.debug("Initialized BaseNode with parent observations.")


class LeafNode(BaseNode):
    """DiFF 树的叶节点。"""
    pass


class InNode(BaseNode):
    """DiFF 树的内部节点。"""

    def __init__(self, X: np.ndarray, height_limit: int, feature_distrib: np.ndarray, sample_size: int, current_height: int) -> None:
        super().__init__(X, current_height, sample_size)
        self.feature_distrib = feature_distrib
        if self.size > 32:
            nbins = int(len(X) / 8) + 2
            weights = np.array([weight_feature(X[:, i], nbins) for i in range(X.shape[1])])
            self.feature_distrib = weights / (weights.sum() + 1e-5)
            logger.debug("Updated feature distribution in InNode.")

        cols = np.arange(X.shape[1], dtype=int)
        self.split_att = rn.choices(cols, weights=self.feature_distrib)[0]
        self.split_value = get_split(X[:, self.split_att])

        left_mask = X[:, self.split_att] <= self.split_value
        right_mask = ~left_mask
        next_height = self.height
        limit_not_reached = height_limit > next_height

        self.left = InNode(X[left_mask], height_limit, self.feature_distrib, sample_size, next_height) if (limit_not_reached and left_mask.sum() > 5 and np.any(X[left_mask].max(axis=0) != X[left_mask].min(axis=0))) else LeafNode(X[left_mask], next_height, sample_size, X)
        self.right = InNode(X[right_mask], height_limit, self.feature_distrib, sample_size, next_height) if (limit_not_reached and right_mask.sum() > 5 and np.any(X[right_mask].max(axis=0) != X[right_mask].min(axis=0))) else LeafNode(X[right_mask], next_height, sample_size, X)

        self.n_nodes += self.left.n_nodes + self.right.n_nodes
        logger.debug(f"Initialized InNode at height {self.height} with {self.n_nodes} nodes.")


class DiFF_Tree:
    """DiFF 树，用于 DiFF_RF 集成。"""

    def __init__(self, height_limit: float) -> None:
        self.height_limit = height_limit
        self.root: Optional[BaseNode] = None

    def fit(self, X: np.ndarray, feature_distrib: np.ndarray) -> BaseNode:
        """拟合树到数据。"""
        try:
            self.root = InNode(X, self.height_limit, feature_distrib, sample_size=len(X), current_height=0)
            logger.info("Fitted DiFF_Tree successfully.")
            return self.root
        except Exception as e:
            logger.error(f"Error fitting DiFF_Tree: {e}")
            raise

    @property
    def n_nodes(self) -> int:
        """返回树中的节点总数。"""
        return self.root.n_nodes if self.root else 0


def walk_tree(
    forest: 'DiFF_TreeEnsemble',
    node: BaseNode,
    tree_idx: int,
    obs_idx: np.ndarray,
    X: np.ndarray,
    feature_distrib: np.ndarray,
    alpha: float = 1e-2
) -> None:
    """递归遍历树以计算路径长度和相似性分数。"""
    if isinstance(node, LeafNode):
        Xnode = X[obs_idx]
        f = ((node.size + 1) / forest.sample_size) / ((1 + len(Xnode)) / forest.Xtest_size)
        if alpha == 0:
            forest.LD[obs_idx, tree_idx] = 0
            forest.LF[obs_idx, tree_idx] = -f
            forest.LDF[obs_idx, tree_idx] = -f
        else:
            z = similarity_score(Xnode, node, alpha)
            forest.LD[obs_idx, tree_idx] = z
            forest.LF[obs_idx, tree_idx] = -f
            forest.LDF[obs_idx, tree_idx] = z * f
        logger.debug(f"Processed LeafNode at tree {tree_idx}, size {node.size}")
    else:
        split_att, split_val = node.split_att, node.split_value
        left_idx = (X[:, split_att] <= split_val) & obs_idx
        right_idx = (X[:, split_att] > split_val) & obs_idx

        walk_tree(forest, node.left, tree_idx, left_idx, X, feature_distrib, alpha)
        walk_tree(forest, node.right, tree_idx, right_idx, X, feature_distrib, alpha)


def create_tree(
    X: np.ndarray,
    feature_distrib: np.ndarray,
    sample_size: int,
    height_limit: float
) -> DiFF_Tree:
    """使用样本创建 DiFF 树。"""
    try:
        sampled_X = X[np.random.choice(len(X), sample_size, replace=False)]
        tree = DiFF_Tree(height_limit=height_limit)
        tree.fit(sampled_X, feature_distrib)
        logger.debug(f"Created tree with sample size {sample_size} and height limit {height_limit}.")
        return tree
    except Exception as e:
        logger.error(f"Error creating tree: {e}")
        raise


class DiFF_TreeEnsemble:
    """DiFF 森林集成，用于异常检测。"""

    def __init__(self, sample_size: int, n_trees: int = 10, alpha: float = 1.0) -> None:
        self.sample_size = sample_size
        self.n_trees = n_trees
        self.alpha = alpha
        self.trees: List[DiFF_Tree] = []
        self.feature_distrib: Optional[np.ndarray] = None
        self.X: Optional[np.ndarray] = None
        self.L, self.LD, self.LF, self.LDF = None, None, None, None
        self.Xtest_size = 0
        self.path_norm_factor = 0.0

        # 初始化随机种子以确保可重复性
        seed = int(time.time())
        np.random.seed(seed)
        rn.seed(seed)
        logger.info("Initialized DiFF_TreeEnsemble.")

    def fit(self, X: np.ndarray, n_jobs: int = 1) -> 'DiFF_TreeEnsemble':
        """拟合集成到训练数据。"""
        self.X = X
        self.path_norm_factor = np.sqrt(len(X))
        self.sample_size = min(self.sample_size, len(X))
        limit_height = np.ceil(np.log2(self.sample_size)).astype(float)

        nbins = int(len(X) / 8) + 2
        self.feature_distrib = np.array([weight_feature(X[:, i], nbins) for i in range(X.shape[1])])
        self.feature_distrib /= (self.feature_distrib.sum() + 1e-5)

        args = [(X, self.feature_distrib, self.sample_size, limit_height) for _ in range(self.n_trees)]

        try:
            with Pool(n_jobs) as pool:
                self.trees = pool.starmap(create_tree, args)
            logger.info(f"Fitted {self.n_trees} trees in the ensemble.")
        except Exception as e:
            logger.error(f"Error fitting trees in the ensemble: {e}")
            raise

        return self

    def walk(self, X: np.ndarray) -> None:
        """遍历所有树以计算测试数据的路径长度和相似性分数。"""
        if not self.trees:
            logger.error("No trees found in the ensemble. Please fit the model first.")
            raise ValueError("The ensemble has not been fitted yet.")

        n_samples = len(X)
        self.LD = np.zeros((n_samples, self.n_trees))
        self.LF = np.zeros((n_samples, self.n_trees))
        self.LDF = np.zeros((n_samples, self.n_trees))

        for tree_idx, tree in enumerate(self.trees):
            walk_tree(
                forest=self,
                node=tree.root,
                tree_idx=tree_idx,
                obs_idx=np.ones(n_samples, dtype=bool),
                X=X,
                feature_distrib=self.feature_distrib,
                alpha=self.alpha
            )
            logger.debug(f"Walked through tree {tree_idx}.")

    def anomaly_score(self, X: np.ndarray, alpha: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """计算测试数据的异常分数。"""
        if alpha is not None:
            self.alpha = alpha

        self.Xtest_size = len(X)
        self.walk(X)

        scD = -self.LD.mean(axis=1)
        scF = self.LF.mean(axis=1)
        scDF = -self.LDF.mean(axis=1)
        logger.info("Computed anomaly scores.")

        return scD, scF, scDF

    def predict_from_anomaly_scores(self, scores: np.ndarray, threshold: float) -> np.ndarray:
        """基于异常分数和阈值生成预测。"""
        predictions = (scores >= threshold).astype(int)
        logger.debug(f"Generated predictions with threshold {threshold}.")
        return predictions

    def predict(
        self,
        X: np.ndarray,
        threshold: float,
        score_type: int = 2
    ) -> np.ndarray:
        """为测试数据生成预测。"""
        if score_type not in {0, 1, 2}:
            logger.error("Invalid score_type provided to predict method.")
            raise ValueError("score_type should be 0 (distance), 1 (frequency), or 2 (collective anomaly score).")

        scores = self.anomaly_score(X)
        prediction_scores = scores[score_type]
        predictions = self.predict_from_anomaly_scores(prediction_scores, threshold)
        logger.info("Generated final predictions.")
        return predictions
