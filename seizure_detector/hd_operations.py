import numpy as np

def generate_item_memory(num_lbp_patterns, num_channels, D=10000):
    """
    生成项目记忆（IM），包含LBP模式和通道的随机向量。

    :param num_lbp_patterns: LBP模式的数量 (例如 2^k)
    :param num_channels: 通道数量
    :param D: 向量维度
    :return: 包含LBP和通道向量的字典
    """
    item_memory = {}
    # 为LBP模式生成向量
    for i in range(num_lbp_patterns):
        item_memory[f'lbp_{i}'] = np.random.randint(2, size=D, dtype=bool)
    # 为通道生成向量
    for i in range(num_channels):
        item_memory[f'channel_{i}'] = np.random.randint(2, size=D, dtype=bool)
    return item_memory

def binding(v1, v2):
    """
    绑定操作（逐位异或）。

    :param v1: 第一个向量
    :param v2: 第二个向量
    :return: 绑定后的向量
    """
    return np.logical_xor(v1, v2)

def bundling(vectors):
    """
    捆绑操作（逐位多数投票），包含了为偶数个向量设计的、确定性的平局处理规则。

    :param vectors: 一个向量列表
    :return: 捆绑后的向量
    """
    if not vectors:
        raise ValueError("Cannot bundle an empty list of vectors.")
    
    vectors_to_process = list(vectors)
    num_vectors = len(vectors_to_process)

    # 1. 检查是否需要处理平局（即输入向量的数量是否为偶数）
    if num_vectors % 2 == 0:
        # 2. 系统地选择前两个向量来创建“平局破坏者”
        vec_a = vectors_to_process[0]
        vec_b = vectors_to_process[1]

        # 3. 通过“绑定”操作(XOR)来创建tie_breaker向量
        tie_breaker_vector = np.logical_xor(vec_a, vec_b)

        # 4. 将这个tie_breaker向量加入到我们的处理列表中
        vectors_to_process.append(tie_breaker_vector)

    # 将向量列表转换为一个二维NumPy数组，然后按列求和
    # axis=0 表示沿着第一个轴（即跨所有向量）进行求和
    sum_vec = np.sum(np.array(vectors_to_process, dtype=int), axis=0)
    final_num_vectors = len(vectors_to_process)

    # 设定阈值。由于现在向量数量一定是奇数，阈值会是一个 .5 的浮点数 (例如 5/2.0=2.5)
    threshold = final_num_vectors / 2.0
    
    bundle_vec = sum_vec > threshold
        
    return bundle_vec

def hamming_distance(v1, v2):
    """
    计算两个二进制向量之间的汉明距离。

    :param v1: 第一个向量
    :param v2: 第二个向量
    :return: 汉明距离
    """
    return np.sum(np.logical_xor(v1, v2))
