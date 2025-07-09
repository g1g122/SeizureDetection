import numpy as np
# 这是一个更高级的numpy工具，专门用于创建滑动窗口视图而无需复制数据
from numpy.lib.stride_tricks import sliding_window_view

def calculate_lbp(signals, k=6):
    """
    为每个通道的信号计算LBP码。

    :param signals: 2D numpy矩阵 (channels, samples)
    :param k: LBP码的位数
    :return: LBP码矩阵 (channels, samples - k)
    """
    num_channels, num_samples = signals.shape
    
    # LBP码会比原始信号短k个点
    lbp_codes = np.zeros((num_channels, num_samples - k), dtype=int)

    for c in range(num_channels):
        for t in range(num_samples - k):
            code = 0
            # 比较后续k个点对的趋势
            for i in range(k):
                # 比较 x(t+i+1) 和 x(t+i)
                if signals[c, t + i + 1] > signals[c, t + i]:
                    code |= (1 << (k - 1 - i))
            lbp_codes[c, t] = code
            
    return lbp_codes

def calculate_lbp_vectorized(signals, k=6):
    """
    使用NumPy向量化高效计算LBP码。

    :param signals: 2D numpy矩阵 (channels, samples)
    :param k: LBP码的位数
    :return: LBP码矩阵 (channels, samples - k)
    """
    # 步骤 1: 一次性计算所有相邻点对的趋势 (上升为True, 否则为False)
    # shape: (channels, samples - 1)
    trends = signals[:, 1:] > signals[:, :-1]
    
    # 步骤 2: 创建一个k位的滑动窗口视图
    # 这个视图让我们能看到在每个时间点的后续k个趋势，而无需循环
    # shape: (channels, samples - k, k)
    trend_windows = sliding_window_view(trends, window_shape=k, axis=1)

    # 步骤 3: 创建二进制权重 (2^(k-1), 2^(k-2), ..., 1)
    # shape: (k,)
    powers_of_2 = 2**(np.arange(k - 1, -1, -1))
    
    # 步骤 4: 使用矩阵乘法将布尔窗口转换为整数LBP码
    # (channels, samples - k, k) @ (k,) -> (channels, samples - k)
    # @ 是numpy中的矩阵乘法运算符
    lbp_codes = trend_windows @ powers_of_2
    
    return lbp_codes