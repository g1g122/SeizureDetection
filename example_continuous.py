import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from seizure_detector import DataHandler, HDCClassifier, PostProcessor, save_results_to_fif

def plot_results(data, sf, detected_starts, detected_durations, ground_truth_start, ground_truth_duration, channel_to_plot=0):
    sns.set_style("ticks")
    sns.set_context("paper", font_scale=1.5)
    signal_color = "#333333"
    truth_color = "#4C72B0"
    detected_color = "#C44E52"
    fig, ax = plt.subplots(figsize=(16, 5))
    time_axis = np.arange(data.shape[1]) / sf
    ax.plot(time_axis, data[channel_to_plot], label=f'iEEG Signal (Ch. {channel_to_plot})', color=signal_color, linewidth=0.8)
    ground_truth_end = ground_truth_start + ground_truth_duration
    ax.axvspan(ground_truth_start, ground_truth_end, color=truth_color, alpha=0.3, label='Ground Truth', lw=0)
    if len(detected_starts) > 0:
        start = detected_starts[0]
        end = start + detected_durations[0]
        ax.axvspan(start, end, color=detected_color, alpha=0.5, label='Detected Seizure', lw=0)
        for i in range(1, len(detected_starts)):
            ax.axvspan(detected_starts[i], detected_starts[i] + detected_durations[i], color=detected_color, alpha=0.5, lw=0)
    ax.set_title('Seizure Detection Performance', fontsize=16, pad=10)
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Amplitude (a.u.)', fontsize=12)
    ax.legend(loc='upper right', frameon=False, fontsize=10)
    ax.set_xlim(0, time_axis[-1])
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.show()

def run_continuous_pipeline():
    """
    演示处理连续长信号的工作流程。
    """
    # --- 1. 准备长信号数据 ---
    print("--- Running Continuous Signal Pipeline ---")
    print("Generating mock data...")
    sf = 512
    num_channels = 4
    duration_min = 10
    num_samples = duration_min * 60 * sf
    
    t = np.linspace(0, duration_min * 60, num_samples)
    mock_data = np.sin(2 * np.pi * 15 * t) + np.random.randn(num_samples) * 0.5
    mock_data = np.tile(mock_data, (num_channels, 1))

    ictal_start_sec = 5 * 60
    ictal_duration_sec = 45
    ictal_start_sample = int(ictal_start_sec * sf)
    ictal_end_sample = int((ictal_start_sec + ictal_duration_sec) * sf)
    
    ictal_signal = np.sin(2 * np.pi * 25 * t[ictal_start_sample:ictal_end_sample]) * 2
    mock_data[:, ictal_start_sample:ictal_end_sample] += ictal_signal 

    # 使用 DataHandler 来管理长信号
    data_handler = DataHandler(mock_data, sf)
    data, sf = data_handler.get_data()
    
    # --- 2. 初始化组件 ---
    classifier = HDCClassifier(num_channels=data.shape[0], tr_percentile=20)
    # 明确指定前后向安全裕度参数，使其易于调整
    post_processor = PostProcessor(tc_threshold=7, pre_margin_sec=1.0, post_margin_sec=5.0)

    # --- 3. 增量式训练 ---
    # 从长信号中提取训练片段
    interictal_period = (60, 90)
    ictal_period = (ictal_start_sec + 5, ictal_start_sec + 35)
    
    interictal_clip = data_handler.get_training_clip(interictal_period)
    ictal_clip = data_handler.get_training_clip(ictal_period)
    
    # 多次调用 train 方法
    classifier.train(interictal_clip, 'interictal', sf)
    classifier.train(ictal_clip, 'ictal', sf)
    
    # 所有发作期片段提供完毕后，计算阈值
    classifier.calculate_tr_threshold([ictal_clip], sf) # 传入所有发作期片段的列表

    # --- 4. 分类长信号 ---
    raw_results = classifier.classify(data, sf)
    
    # --- 5. 后处理 ---
    # 将自动计算的阈值传递给后处理器
    post_processor.tr_threshold = classifier.tr_threshold 
    final_decisions = post_processor.smooth_labels(raw_results, tr_threshold=post_processor.tr_threshold)
    start_times, durations = post_processor.aggregate_events(final_decisions)

    # --- 6. 输出和可视化 ---
    print("\n--- Seizure Detection Results ---")
    if len(start_times) > 0:
        for start, duration in zip(start_times, durations):
            print(f"Detected Seizure -> Start: {start:.2f} s, Duration: {duration:.2f} s")
    else:
        print("No seizures detected.")
    print("---------------------------------")
    
    plot_results(data, sf, start_times, durations, ictal_start_sec, ictal_duration_sec)

    # --- 7. 保存结果到 .fif 文件 ---
    print("\n--- Saving results to .fif file ---")
    output_filename = "mock_ieeg.fif"
    save_results_to_fif(data, sf, start_times, durations, output_filename)


if __name__ == '__main__':
    run_continuous_pipeline() 