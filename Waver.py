# 这个是最新的代码
from typing import Union


import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, savgol_filter

class Waver:
    
    peak_dtype = np.dtype([
        ("position", int),
        ("height", float),
        ("integral", float),
        ("edge_start", int),
        ("edge_end", int),
    ])
    
    

    def __init__(self, waveform: np.ndarray):
        self.waveform = waveform  # 原始波形
        self.timestamp=waveform
        self.filtered_waveform = None  # 滤波后的波形
        self.filtered_waveform_diff = None  # 滤波后的波形的一阶导数
        # 定义峰值信息的结构
        self.peaks_info = None  # 存储峰值信息（位置、峰高、积分）

    def waveform_filter(self, lowcut=0.1, highcut=0.5, fs=1.0, order=4, filter="SG"):
        """
        对波形进行带通滤波或Savitzky-Golay滤波
        :param lowcut: 低频截止
        :param highcut: 高频截止
        :param fs: 采样率
        :param order: 滤波器阶数
        :param filter: 滤波类型，"BW"为带通滤波器，"SG"为Savitzky-Golay滤波器
        """
        if filter == "BW":
            # 创建带通滤波器（Butterworth滤波器）
            nyquist = 0.5 * fs
            low = lowcut / nyquist
            high = highcut / nyquist

            b, a = butter(order, [low, high], btype="band")
            # 应用带通滤波器
            self.filtered_waveform = filtfilt(b, a, self.waveform)

        elif filter == "SG":
            # 使用Savitzky-Golay滤波器
            window_size = 10
            poly_order = 2
            self.filtered_waveform = savgol_filter(self.waveform, window_size, poly_order)

        else:
            raise ValueError("filter must be 'BW' or 'SG'. You can add new ones by yourself.")

        self.filtered_waveform_diff = np.diff(self.filtered_waveform)

    def find_peaks(self, height=None, distance=None, threshold=None, prominence=None):
        """
        寻找波形中的峰值，并计算峰高和积分等信息
        :param height: 峰值的最小高度
        :param distance: 峰值之间的最小距离
        """
        # 使用 find_peaks 查找峰值索引和属性（如峰高）

        if self.filtered_waveform is None:
            self.filtered_waveform()

        # peaks, properties = find_peaks(
        #     waveform_used,
        #     height=height,
        #     distance=distance,
        #     threshold=threshold,
        #     prominence=prominence,
        # )
        filtered_waveform_diff = -np.diff(self.get_filtered_waveform())

        peaks, properties = find_peaks(
            filtered_waveform_diff, height=30, distance=2, prominence=0.7, width=4
        )

        # left_lens = peaks - properties["left_ips"]
        # right_lens = properties["right_ips"] - peaks

        # starts = int(np.round(properties["left_ips"]))
        # ends = int(np.round(properties["right_ips"]))
        starts = [int(np.round(start)) for start in properties["left_ips"]]
        ends = [int(np.round(end)) for end in properties["right_ips"]]
        peak_heights = self._peak_height_from_diff(starts, ends)

        peak_integrals = np.array([None for p in peaks])

        # 存储峰值信息（位置、峰高、积分）
        self.peaks_info = np.array(
            [
                (
                    peaks[i],
                    peak_heights[i],
                    peak_integrals[i],
                    properties["left_ips"][i],
                    properties["right_ips"][i],
                )
                for i in range(len(peaks))
            ],
            dtype=self.peak_dtype,
        )
        self.proeperties = properties
        # 更新 peaks 索引
        self.peaks = peaks

    def get_peaks(self):
        """
        获取当前的峰值信息（位置、峰高、积分等）
        :return: 包含峰值信息的数组
        """
        if self.peaks_info is None:
            self.find_peaks()
        return self.peaks_info

    def _peak_height_from_diff(self, starts: Union[int, list[int]], ends: Union[int, list[int]]):
        peak_heights = []

        for start, end in zip(starts, ends):
            # 计算区间内的波形高度

            peak_height = np.sum(np.diff(-self.get_filtered_waveform())[start:end])
            peak_heights.append(peak_height)

        return np.array(peak_heights)

    def get_filtered_waveform(self,**kwargs):
        """
        获取滤波后的波形
        :return: 滤波后的波形
        """
        if self.filtered_waveform is None:
            self.waveform_filter(kwargs)
        return self.filtered_waveform