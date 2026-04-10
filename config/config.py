import os
import subprocess


class Config:
    # 优化器学习率
    LEARNING_RATE = 1e-3
    # 每个 batch 中包含的 mesh 数量
    BATCH_SIZE = 1
    # 最大训练轮数
    MAX_EPOCHS = 5000
    # 最大样本数量
    OBJ_LIMIT = 1
    DATASET_PERCENT = 100
    VAL_EVERY_N_EPOCHS = 10
    TRAIN_VIS_EVERY_N_EPOCHS = 10
    # 早停耐心值：如果验证集 loss 在多少个 epoch 内没有下降，就提前停止。如果设为足够大（如 500），等同于关闭早停
    EARLY_STOP_PATIENCE = 50
    # 权重衰减系数
    WEIGHT_DECAY = 1e-4

    # 训练方法
    # bipartite：匈牙利
    # anchor: 类似 YOLO
    # direct: 直接预测
    # modal_anchor: 模态锚点预测, 见current_methods.md
    # PREDICTION_MODE = "anchor" 
    PREDICTION_MODE = "modal_anchor"

    # 固定分箱
    USE_MODAL_BINS = True
    
    # 训练设备类型，可选 cpu / cuda / mps
    DEVICE = "cuda"
    
    # --- GPU 配置 ---
    # 模式1：手动指定 GPU (例如 [0, 1])。如果不为空，则直接使用指定的 GPU
    GPU_IDS = [3]
    # 模式2：自动寻找空闲 GPU。如果 AUTO_FIND_GPUS 为 True，则忽略 GPU_IDS，自动寻找 DEVICES 个空闲 GPU
    AUTO_FIND_GPUS = False
    # 使用的设备数量（手动指定时会被自动同步，自动寻找时用于指定寻找的数量）
    DEVICES = 1

    # 数据根目录
    DATA_DIR = "/mnt/GIL-NFS/xuchenxi/vv++/objaverse-gen"
    # 缓存目录
    CACHE_DIR = "/mnt/GIL-NFS/xuchenxi/vvpp-project/.cache"
    
    # DataLoader worker 数量
    NUM_WORKERS = 8

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

    def __init__(self):
        if self.DEVICE == "cuda":
            if self.AUTO_FIND_GPUS:
                self.GPU_IDS = self._get_free_gpus(self.DEVICES)
            else:
                self.DEVICES = len(self.GPU_IDS)
            
            # 自动设置环境变量，确保后续代码(如 PyTorch)只使用被选中的 GPU
            if self.GPU_IDS:
                os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.GPU_IDS))
        else:
            self.GPU_IDS = []

    def _get_free_gpus(self, num_gpus):
        """
        使用 nvidia-smi 自动寻找显存占用最少的 N 个空闲 GPU
        """
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, check=True
            )
            lines = result.stdout.strip().split('\n')
            gpu_memory = []
            for line in lines:
                if line.strip():
                    idx, mem = line.split(',')
                    gpu_memory.append((int(idx.strip()), int(mem.strip())))
            
            # 按照显存占用升序排序
            gpu_memory.sort(key=lambda x: x[1])
            # 返回最空闲的 num_gpus 个 GPU 索引
            return [x[0] for x in gpu_memory[:num_gpus]]
        except Exception as e:
            print(f"Warning: Failed to auto-find GPUs via nvidia-smi: {e}")
            # 失败时降级：默认返回前 num_gpus 个 GPU
            return list(range(num_gpus))


cfg = Config()
