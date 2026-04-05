import os


class Config:
    # 优化器学习率
    LEARNING_RATE = 1e-3
    # 每个 batch 中包含的 mesh 数量
    BATCH_SIZE = 2
    # 最大训练轮数
    MAX_EPOCHS = 100
    # 权重衰减系数
    WEIGHT_DECAY = 1e-4
    # 训练设备类型，可选 cpu / cuda / mps
    DEVICE = "cpu"
    # 使用的设备数量
    DEVICES = 1

    # 数据根目录
    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "vv++test")
    # DataLoader worker 数量
    NUM_WORKERS = 4

    # 隐藏层特征维度
    HIDDEN_DIM = 256
    # 输出特征维度
    OUTPUT_DIM = 256

    # 音频采样率
    SAMPLE_RATE = 16000
    # 拉普拉斯特征模态数量
    N_EIGENMODES = 64
    # 八叉树最大深度
    OCTREE_DEPTH = 6
    # 八叉树完整展开深度
    OCTREE_FULL_DEPTH = 4
    # 八叉树是否仅保留非空节点
    OCTREE_NEMPTY = False


cfg = Config()
