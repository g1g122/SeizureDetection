"""
Seizure Detector Package
========================

这是一个基于超维计算（HDC）理论，用于检测癫痫事件的Python包。
"""

# 从子模块中导入核心类，以便用户可以直接从包顶层访问
from .data_handler import DataHandler
from .hdc_classifier import HDCClassifier
from .post_processor import PostProcessor
from .export import save_results_to_fif

# 定义包的公开API，当其他开发者使用 `from seizure_detector import *` 时，
# 只会导入下面列表中的名称。
__all__ = [
    'DataHandler',
    'HDCClassifier',
    'PostProcessor',
    'save_results_to_fif'
]
