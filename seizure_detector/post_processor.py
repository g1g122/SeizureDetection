import numpy as np

class PostProcessor:
    """
    对分类器的原始输出进行后处理，以平滑结果并聚合事件。
    """
    def __init__(self, vote_window_sec=5.0, step_sec=0.5, tc_threshold=7, pre_margin_sec=1.0, post_margin_sec=5.0):
        """
        :param vote_window_sec: 投票窗口大小（秒）
        :param step_sec: 分类标签流的时间步长（秒）
        :param tc_threshold: 时间一致性阈值，判定为“SEIZURE”的“Ictal”标签数量
        :param pre_margin_sec: 在检测到的事件开始前添加的前向安全裕度（秒）
        :param post_margin_sec: 在检测到的事件结束后添加的后向安全裕度（秒）
        """
        self.vote_window_labels = int(vote_window_sec / step_sec)
        self.step_sec = step_sec
        self.tc_threshold = tc_threshold
        self.pre_margin_sec = pre_margin_sec
        self.post_margin_sec = post_margin_sec
        self.tr_threshold = None # 初始化置信度阈值

    def smooth_labels(self, classification_results, tr_threshold):
        """
        使用双阈值（时间一致性 + 置信度）机制平滑标签流。

        :param classification_results: (标签, 置信度)元组的列表
        :param tr_threshold: 平均置信度阈值
        :return: 平滑后的决策列表 ["SEIZURE" or "NON-SEIZURE"]
        """
        num_results = len(classification_results)
        decisions = []
        
        for i in range(num_results - self.vote_window_labels + 1):
            window = classification_results[i : i + self.vote_window_labels]
            
            # 1. 时间一致性检查 (tc)
            ictal_results = [res for res in window if res[0] == "Ictal"]
            ictal_count = len(ictal_results)
            
            if ictal_count >= self.tc_threshold:
                # 2. 置信度检查 (tr)
                confidences = [res[1] for res in ictal_results]
                avg_confidence = np.mean(confidences)
                
                if avg_confidence > tr_threshold:
                    decisions.append("SEIZURE")
                else:
                    decisions.append("NON-SEIZURE")
            else:
                decisions.append("NON-SEIZURE")
                
        return decisions

    def process_clip_results(self, classification_results):
        """
        对单个片段的分类结果进行最终裁决。

        :param classification_results: 来自 classify_clip 的(标签, 置信度)元组列表
        :return: "SEIZURE" 或 "NON-SEIZURE"
        """
        if self.tr_threshold is None:
            raise RuntimeError("TR threshold has not been set. Cannot process clip results.")
            
        # 1. 时间一致性检查 (tc)
        ictal_results = [res for res in classification_results if res[0] == "Ictal"]
        ictal_count = len(ictal_results)

        if ictal_count >= self.tc_threshold:
            # 2. 置信度检查 (tr)
            confidences = [res[1] for res in ictal_results]
            avg_confidence = np.mean(confidences) if confidences else 0
            
            if avg_confidence > self.tr_threshold:
                return "SEIZURE"
        
        return "NON-SEIZURE"

    def aggregate_events(self, decisions):
        """
        从决策流中聚合癫痫事件，并应用安全裕度及合并重叠事件。
        """
        # 步骤 1: 初始事件检测
        seizure_start_times = []
        seizure_durations = []
        in_seizure = False
        start_time = 0

        for i, decision in enumerate(decisions):
            current_time = i * self.step_sec
            if not in_seizure and decision == "SEIZURE":
                in_seizure = True
                start_time = current_time
            elif in_seizure and decision == "NON-SEIZURE":
                in_seizure = False
                duration = current_time - start_time
                seizure_start_times.append(start_time)
                seizure_durations.append(duration)
        
        # 处理在决策流末尾仍未结束的癫痫事件
        if in_seizure:
            duration = (len(decisions) * self.step_sec) - start_time
            seizure_start_times.append(start_time)
            seizure_durations.append(duration)

        if not seizure_start_times:
            return np.array([]), np.array([])

        # 步骤 2: 应用安全裕度，并转换为 (开始, 结束) 区间
        extended_events = []
        for start, duration in zip(seizure_start_times, seizure_durations):
            new_start = max(0, start - self.pre_margin_sec)
            new_end = start + duration + self.post_margin_sec
            extended_events.append([new_start, new_end])

        # 步骤 3: 融合重叠的事件区间
        # 首先按开始时间排序
        extended_events.sort(key=lambda x: x[0])
        
        if not extended_events:
            return np.array([]), np.array([]) # 增加健壮性检查

        merged_events = [extended_events[0]]
        for current_event in extended_events[1:]:
            # 直接与 merged_events 的最后一个元素进行比较和修改，意图更明确
            if current_event[0] <= merged_events[-1][1]:
                # 有重叠，直接更新最后一个元素的结束时间
                merged_events[-1][1] = max(merged_events[-1][1], current_event[1])
            else:
                # 无重叠，将当前事件添加为新事件
                merged_events.append(current_event)

        # 步骤 4: 将融合后的区间转换回 (开始时间, 持续时长) 格式
        final_starts = np.array([event[0] for event in merged_events])
        final_durations = np.array([event[1] - event[0] for event in merged_events])

        return final_starts, final_durations
