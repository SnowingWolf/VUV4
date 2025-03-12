from typing import List, Optional, Union

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, savgol_filter


class Waver:
    """
    Waver类用于处理波形数据，包括滤波和峰值检测。
    """

    peak_dtype = np.dtype([
        ("position", int),
        ("height", float),
        ("integral", float),
        ("edge_start", int),
        ("edge_end", int),
    ])

    def __init__(self, waveform: np.ndarray, timestamp=None):
        """
        初始化Waver对象。

        Args:
            waveform (np.ndarray): 原始波形数据。
        """
        self.timestamp = timestamp  # 时间戳
        self.waveform = waveform  # 原始波形
        self.filtered_waveform = None  # 滤波后的波形
        self.filtered_waveform_diff = None  # 滤波后的波形的一阶导数
        self.peaks_info = None  # 峰值信息数组
        self.properties = None  # 峰值检测的属性

    def waveform_filter(
        self,
        lowcut: float = 0.1,
        highcut: float = 0.5,
        fs: float = 1.0,
        order: int = 4,
        filter_type: str = "SG",
        window_length: int = 10,
        poly_order: int = 2,
    ):
        """
        对波形进行滤波，支持Butterworth带通滤波器和Savitzky-Golay滤波器。

        Args:
            lowcut (float, optional): 低频截止频率。默认为0.1。
            highcut (float, optional): 高频截止频率。默认为0.5。
            fs (float, optional): 采样率。默认为1.0。
            order (int, optional): 滤波器阶数。默认为4。
            filter_type (str, optional): 滤波器类型，"BW"或"SG"。默认为"BW"。
            window_length (int, optional): SG滤波器的窗口长度。默认为10。
            poly_order (int, optional): SG滤波器的多项式阶数。默认为2。
        """
        if filter_type == "BW":
            nyquist = 0.5 * fs
            low = lowcut / nyquist
            high = highcut / nyquist
            b, a = butter(order, [low, high], btype="band")
            self.filtered_waveform = filtfilt(b, a, self.waveform)
        elif filter_type == "SG":
            self.filtered_waveform = savgol_filter(self.waveform, window_length, poly_order)
        else:
            raise ValueError("filter_type must be 'BW' or 'SG'.")
        self.filtered_waveform_diff = np.diff(self.filtered_waveform)

    def find_peaks(
        self,
        height: Optional[float] = None,
        distance: Optional[int] = None,
        threshold: Optional[float] = None,
        prominence: Optional[float] = None,
        width: Optional[float] = None,
    ):
        """
        寻找波形中的峰值，并计算峰高和积分等信息。

        Args:
            height (Optional[float], optional): 峰值的最小高度。默认为None。
            distance (Optional[int], optional): 峰值之间的最小距离。默认为None。
            threshold (Optional[float], optional): 峰值的最小阈值。默认为None。
            prominence (Optional[float], optional): 峰值的最小突出度。默认为None。
            width (Optional[float], optional): 峰值的最小宽度。默认为None。
        """
        if self.filtered_waveform is None:
            self.waveform_filter()  # 使用默认参数滤波

        waveform_used = self.get_filtered_waveform()

        peaks, properties = find_peaks(
            waveform_used,
            height=height,
            distance=distance,
            threshold=threshold,
            prominence=prominence,
            width=width,
        )

        starts = [int(np.round(properties["left_ips"][i])) for i in range(len(peaks))]
        ends = [int(np.round(properties["right_ips"][i])) for i in range(len(peaks))]
        peak_heights = self._peak_height_from_diff(starts, ends)
        peak_integrals = np.array([
            self._calculate_peak_integral(start, end) for start, end in zip(starts, ends)
        ])

        self.peaks_info = np.array(
            [
                (peaks[i], peak_heights[i], peak_integrals[i], starts[i], ends[i])
                for i in range(len(peaks))
            ],
            dtype=self.peak_dtype,
        )
        self.properties = properties

    def get_peaks(self) -> np.ndarray:
        """
        获取当前的峰值信息。

        Returns:
            np.ndarray: 包含峰值信息的数组。
        """
        if self.peaks_info is None:
            self.find_peaks()
        return self.peaks_info

    def _peak_height_from_diff(
        self, starts: Union[int, List[int]], ends: Union[int, List[int]]
    ) -> np.ndarray:
        """
        从差分数组计算峰值的高度。

        Args:
            starts (Union[int, List[int]]): 巅峰起始索引。
            ends (Union[int, List[int]]): 巅峰结束索引。

        Returns:
            np.ndarray: 峰值的高度数组。
        """
        if not isinstance(starts, list):
            starts = [starts]
            ends = [ends]

        peak_heights = []
        diff_waveform = np.diff(self.get_filtered_waveform())

        for start, end in zip(starts, ends):
            if start >= end:
                peak_heights.append(0.0)
                continue
            integral = np.sum(diff_waveform[start:end])
            peak_heights.append(abs(integral))

        return np.array(peak_heights)

    def _calculate_peak_integral(self, start: int, end: int) -> float:
        """
        计算某个区间的积分。

        Args:
            start (int): 区间起始索引。
            end (int): 区间结束索引。

        Returns:
            float: 积分值。
        """
        if start >= end:
            return 0.0
        return np.trapz(self.get_filtered_waveform()[start:end], x=None)

    def get_filtered_waveform(self, **args) -> np.ndarray:
        """
        获取滤波后的波形。

        Returns:
            np.ndarray: 滤波后的波形。
        """
        if self.filtered_waveform is None:
            self.waveform_filter(args)  # 使用默认参数滤波
        return self.filtered_waveform
