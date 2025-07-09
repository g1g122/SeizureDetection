import numpy as np
from . import hd_operations

class HDEncoder:
    """
    将LBP特征编码为超维向量（H向量）。
    """
    def __init__(self, item_memory, D=10000):
        """
        :param item_memory: 预先生成的项目记忆
        :param D: 向量维度
        """
        self.item_memory = item_memory
        self.D = D

    def encode_window(self, lbp_window):
        """
        将一个窗口内的LBP码编码为单个H向量。

        :param lbp_window: LBP码窗口 (channels, window_samples)
        :return: H向量
        """
        num_channels, window_samples = lbp_window.shape
        s_vectors = []

        # 空间捆绑
        for t in range(window_samples):
            bound_vectors = []
            for c in range(num_channels):
                lbp_code = lbp_window[c, t]
                lbp_vec = self.item_memory[f'lbp_{lbp_code}']
                channel_vec = self.item_memory[f'channel_{c}']
                
                bound_vec = hd_operations.binding(lbp_vec, channel_vec)
                bound_vectors.append(bound_vec)
            
            s_vec = hd_operations.bundling(bound_vectors)
            s_vectors.append(s_vec)

        # 时间捆绑
        h_vector = hd_operations.bundling(s_vectors)
        return h_vector
