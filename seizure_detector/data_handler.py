import numpy as np

class DataHandler:
    """
    用于加载和管理iEEG数据的类。
    当前实现为占位符，直接使用numpy数组。
    """
    def __init__(self, data, sf):
        """
        :param data: numpy数组 (channels, samples)
        :param sf: 采样频率
        """
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise ValueError("Data must be a 2D numpy array (channels, samples).")
        if not isinstance(sf, (int, float)) or sf <= 0:
            raise ValueError("Sampling frequency (sf) must be a positive number.")
            
        self.data = data
        self.sf = sf
        self.num_channels, self.num_samples = data.shape

    def get_data(self):
        return self.data, self.sf
    
    def get_training_clip(self, period):
        """
        根据给定的时间段（秒）从长信号中提取一个数据片段。

        :param period: (开始秒, 结束秒) 的元组
        :return: 对应的数据片段 (numpy array)
        """
        start_sample = int(period[0] * self.sf)
        end_sample = int(period[1] * self.sf)
        
        if start_sample < 0 or end_sample > self.num_samples:
            raise ValueError("提供的训练时间段超出了数据范围。")
            
        return self.data[:, start_sample:end_sample]
