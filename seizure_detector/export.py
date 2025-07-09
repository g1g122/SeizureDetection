import numpy as np

def save_results_to_fif(data, sf, start_times, durations, output_path, ch_names=None):
    """
    将连续信号和检测到的癫痫事件打包为MNE实例，并保存为 .fif 文件。

    :param data: 原始信号数据 (channels, samples)
    :param sf: 采样频率
    :param start_times: 检测到的发作开始时间列表 (秒)
    :param durations: 检测到的发作持续时间列表 (秒)
    :param output_path: 输出的 .fif 文件路径
    :param ch_names: 通道名称列表 (可选)
    """
    try:
        import mne
    except ImportError:
        print("MNE-Python is not installed. Please install it using: pip install mne")
        return

    print(f"Saving results to {output_path}...")

    # 1. 创建通道信息
    num_channels = data.shape[0]
    if ch_names is None:
        ch_names = [f'Chan {i+1}' for i in range(num_channels)]
    elif len(ch_names) != num_channels:
        raise ValueError("The number of channel names does not match the number of channels in the data.")
    
    # MNE 要求通道类型为 'eeg', 'meg', 'seeg', 'ecog' 等
    # 这里我们统一使用 'seeg' (Stereotactic EEG) 作为示例
    ch_types = ['seeg'] * num_channels
    info = mne.create_info(ch_names=ch_names, sfreq=sf, ch_types=ch_types)

    # 2. 创建 MNE Raw 对象
    raw = mne.io.RawArray(data, info)

    # 3. 创建 Annotations 对象
    # MNE 的 Annotations 需要 onset, duration, 和 description
    if len(start_times) > 0:
        annotations = mne.Annotations(onset=start_times,
                                      duration=durations,
                                      description=['Seizure'] * len(start_times))
        # 将标注添加到 Raw 对象
        raw.set_annotations(annotations)
    else:
        print("No seizures detected, saving file without annotations.")

    # 4. 保存为 .fif 文件
    # overwrite=True 允许覆盖已存在的文件
    raw.save(output_path, overwrite=True)
    print("File saved successfully.") 