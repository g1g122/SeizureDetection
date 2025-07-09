import numpy as np
from . import lbp_feature_extractor
from . import hd_operations
from .hd_encoder import HDEncoder

class HDCClassifier:
    """
    基于超维计算的分类器，用于癫痫检测。
    """
    def __init__(self, num_channels, D=10000, k=6, window_sec=1.0, step_sec=0.5, tr_percentile=20):
        """
        :param num_channels: 信号的通道数
        :param D: 超维向量的维度
        :param k: LBP码的位数
        :param window_sec: 时间窗口大小（秒）
        :param step_sec: 滑动窗口步长（秒）
        :param tr_percentile: 用于自动计算置信度阈值的百分位数
        """
        self.num_channels = num_channels
        self.D = D
        self.k = k
        self.window_sec = window_sec
        self.step_sec = step_sec
        self.tr_percentile = tr_percentile
        
        # 1. 生成项目记忆 (IM)
        num_lbp_patterns = 2**k
        self.item_memory = hd_operations.generate_item_memory(num_lbp_patterns, num_channels, D)
        
        # 2. 初始化编码器
        self.encoder = HDEncoder(self.item_memory, D)
        
        # 3. 初始化原型向量和阈值
        self.ictal_prototype = None
        self.interictal_prototype = None
        self.tr_threshold = None

    def _update_prototype(self, data_clip, sf, existing_prototype):
        """
        使用新的数据片段来更新（或创建）一个原型向量。
        """
        # 为新片段计算H向量
        lbp_codes = lbp_feature_extractor.calculate_lbp_vectorized(data_clip, self.k)
        window_samples = int(self.window_sec * sf)
        step_samples = int(self.step_sec * sf)
        
        new_h_vectors = []
        num_windows = (lbp_codes.shape[1] - window_samples) // step_samples + 1
        for i in range(num_windows):
            start = i * step_samples
            end = start + window_samples
            lbp_window = lbp_codes[:, start:end]
            new_h_vectors.append(self.encoder.encode_window(lbp_window))

        if not new_h_vectors:
            return existing_prototype # 如果片段太短，不更新

        # 如果原型已存在，将新旧向量捆绑；否则，创建新原型
        if existing_prototype is not None:
            updated_prototype = hd_operations.bundling([existing_prototype] + new_h_vectors)
        else:
            updated_prototype = hd_operations.bundling(new_h_vectors)
            
        return updated_prototype

    def train(self, data_clip, label, sf):
        """
        增量式训练：使用单个数据片段及其标签来更新原型。

        :param data_clip: 训练数据片段 (numpy array)
        :param label: 'ictal' 或 'interictal'
        :param sf: 采样频率
        """
        if label == 'ictal':
            print("Updating Ictal prototype...")
            self.ictal_prototype = self._update_prototype(data_clip, sf, self.ictal_prototype)
        elif label == 'interictal':
            print("Updating Interictal prototype...")
            self.interictal_prototype = self._update_prototype(data_clip, sf, self.interictal_prototype)
        else:
            raise ValueError("Label must be 'ictal' or 'interictal'.")

    def calculate_tr_threshold(self, ictal_training_clips, sf):
        """
        在所有发作期训练片段上计算置信度分数，以确定阈值。
        应在所有 'ictal' 样本训练完毕后调用。

        :param ictal_training_clips: 一个包含所有发作期训练片段的列表
        :param sf: 采样频率
        """
        if self.ictal_prototype is None or self.interictal_prototype is None:
            raise RuntimeError("Both prototypes must be trained before calculating threshold.")

        print(f"Calculating confidence threshold (tr) using {self.tr_percentile}th percentile...")
        
        ictal_confidences = []
        for clip in ictal_training_clips:
            results = self.classify_clip(clip, sf)
            confidences = [conf for label, conf in results if label == 'Ictal']
            ictal_confidences.extend(confidences)

        if not ictal_confidences:
            raise ValueError("Could not compute any confidence scores from the ictal training data.")

        self.tr_threshold = np.percentile(ictal_confidences, self.tr_percentile)
        print(f"Confidence threshold (tr) automatically set to: {self.tr_threshold:.2f}")

    def classify_clip(self, data_clip, sf):
        """
        对单个数据片段进行分类，返回每个子窗口的结果。
        """
        if self.ictal_prototype is None or self.interictal_prototype is None:
            raise RuntimeError("Classifier has not been trained yet.")

        lbp_codes = lbp_feature_extractor.calculate_lbp_vectorized(data_clip, self.k)
        window_samples = int(self.window_sec * sf)
        step_samples = int(self.step_sec * sf)
        
        results = []
        num_windows = (lbp_codes.shape[1] - window_samples) // step_samples + 1
        
        for i in range(num_windows):
            start = i * step_samples
            end = start + window_samples
            lbp_window = lbp_codes[:, start:end]
            h_query = self.encoder.encode_window(lbp_window)
            
            dist_to_ictal = hd_operations.hamming_distance(h_query, self.ictal_prototype)
            dist_to_interictal = hd_operations.hamming_distance(h_query, self.interictal_prototype)
            
            confidence = abs(dist_to_ictal - dist_to_interictal)
            if dist_to_ictal < dist_to_interictal:
                results.append(("Ictal", confidence))
            else:
                results.append(("Interictal", confidence))
        return results

    def classify(self, data, sf):
        """
        对整个连续数据集进行分类 (长信号模式)。
        """
        print("Classifying long continuous signal...")
        return self.classify_clip(data, sf)