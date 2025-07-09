import numpy as np
from seizure_detector import HDCClassifier, PostProcessor

def create_mock_clip(sf, duration_sec, is_ictal):
    """一个简单的函数，用于创建模拟的数据片段。"""
    num_samples = int(duration_sec * sf)
    t = np.linspace(0, duration_sec, num_samples)
    # 基础信号
    clip = np.sin(2 * np.pi * 15 * t) + np.random.randn(num_samples) * 0.5
    if is_ictal:
        # 叠加一个更高频、更高幅度的信号来模拟发作
        ictal_signal = np.sin(2 * np.pi * 25 * t) * 2
        clip += ictal_signal
    # 扩展为4通道
    return np.tile(clip, (4, 1))

def run_clips_pipeline():
    """
    演示处理离散数据片段（clips）的工作流程。
    """
    print("\n--- Running Clips-based Pipeline ---")
    sf = 512
    num_channels = 4

    # --- 1. 准备训练用的数据片段 ---
    print("Generating training clips...")
    # 假设我们有多个发作期和发作间期的片段
    ictal_clips_for_training = [
        create_mock_clip(sf, 30, is_ictal=True),
        create_mock_clip(sf, 25, is_ictal=True)
    ]
    interictal_clips_for_training = [
        create_mock_clip(sf, 40, is_ictal=False),
        create_mock_clip(sf, 35, is_ictal=False)
    ]

    # --- 2. 初始化组件 ---
    classifier = HDCClassifier(num_channels=num_channels, tr_percentile=20)
    post_processor = PostProcessor(tc_threshold=7)

    # --- 3. 增量式训练 ---
    print("Training incrementally...")
    for clip in ictal_clips_for_training:
        classifier.train(clip, 'ictal', sf)
    for clip in interictal_clips_for_training:
        classifier.train(clip, 'interictal', sf)
        
    # 所有发作期片段提供完毕后，计算阈值
    classifier.calculate_tr_threshold(ictal_clips_for_training, sf)

    # --- 4. 准备测试用的数据片段 ---
    print("\nGenerating test clips...")
    test_clips = {
        "Test Clip 1 (Ictal)": create_mock_clip(sf, 10, is_ictal=True),
        "Test Clip 2 (Interictal)": create_mock_clip(sf, 12, is_ictal=False),
        "Test Clip 3 (Ictal - borderline)": create_mock_clip(sf, 8, is_ictal=True) * 0.6, # 信号弱一点
        "Test Clip 4 (Interictal with noise)": create_mock_clip(sf, 15, is_ictal=False) + np.random.randn(4, 15*sf)*0.5
    }

    # --- 5. 对每个测试片段进行分类和裁决 ---
    print("Classifying test clips...")
    # 将自动计算的阈值传递给后处理器
    post_processor.tr_threshold = classifier.tr_threshold

    for name, clip in test_clips.items():
        # 分类片段
        clip_results = classifier.classify_clip(clip, sf)
        # 对结果进行最终裁决
        final_decision = post_processor.process_clip_results(clip_results)
        print(f"  - {name}: Final Decision -> {final_decision}")


if __name__ == '__main__':
    # 演示两种模式
    # run_continuous_pipeline() # This function is not defined in the original file, so it's commented out.
    run_clips_pipeline() 