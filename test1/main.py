import numpy as np
from scipy.stats import rayleigh, rice, nakagami
from scipy.signal import coherence
import matplotlib.pyplot as plt

import matplotlib

# 设置中文字体和支持负号
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 显示中文
matplotlib.rcParams['axes.unicode_minus'] = False             # 支持负号

class WirelessChannel:
    def __init__(self, fc=900e6, hb=50, hm=1.5, d0=100, area='urban'):
        """
        初始化无线信道参数
        :param fc: 载波频率 (Hz)
        :param hb: 基站天线高度 (m)
        :param hm: 移动台天线高度 (m)
        :param d0: 参考距离 (m)
        :param area: 区域类型 (urban/suburban/rural)
        """
        self.fc = fc
        self.hb = hb
        self.hm = hm
        self.d0 = d0
        self.area = area.lower()
        self.c = 3e8  # 光速
        
    def hata_path_loss(self, d):
        """
        Hata模型路径损耗计算
        :param d: 传播距离 (m)
        :return: 路径损耗 (dB)
        """
        if self.area == 'urban':
            a_hm = (1.1 * np.log10(self.fc) - 0.7) * self.hm - (1.56 * np.log10(self.fc) - 0.8)
        elif self.area == 'suburban':
            a_hm = 0
        else:  # rural
            a_hm = 0
        
        L = 46.3 + 33.9 * np.log10(self.fc) - 13.82 * np.log10(self.hb) - a_hm + (44.9 - 6.55 * np.log10(self.hb)) * np.log10(d/self.d0)
        return L
    
    def shadow_fading(self, sigma=8, size=1):
        """
        阴影衰落生成（对数正态分布）
        :param sigma: 标准差 (dB)
        :param size: 生成样本数
        :return: 阴影衰落值 (dB)
        """
        return np.random.normal(0, sigma, size)
    
    def jakes_fading(self, N=100, theta=None, fd=100, model='rayleigh', K=10, m=2):
        """
        Jakes模型快衰落生成
        :param N: 多径数目
        :param theta: 到达角 (rad), 随机生成时设为None
        :param fd: 最大多普勒频移 (Hz)
        :param model: 衰落模型 (rayleigh/rice/nakagami)
        :param K: 莱斯因子 (仅莱斯模型)
        :param m: Nakagami形状参数 (仅Nakagami模型)
        :return: 复衰落系数
        """
        if theta is None:
            theta = 2 * np.pi * np.random.rand(N)  # 均匀分布到达角
        
        t = np.linspace(0, 0.01, 1000)  # 时间序列
        
        # 添加维度适配广播运算，将theta从(N,)变为(N,1)
        theta = theta[:, np.newaxis]  
        t = t[np.newaxis, :]  # 将t从(1000,)变为(1, 1000)
        
        phase = 2 * np.pi * fd * t * np.cos(theta) + np.random.uniform(0, 2*np.pi, (N, 1))  # 生成(N,1000)的相位矩阵
        
        alpha = np.sqrt(2/N) * np.cos(phase)
        beta = np.sqrt(2/N) * np.sin(phase)
        fading_baseband = np.sum(alpha + 1j*beta, axis=0)  # 沿多径维度求和，得到(1000,)的基带信号
        
        # 转换为对应衰落模型（保持幅度和相位独立生成）
        if model == 'rayleigh':
            magnitude = rayleigh.rvs(scale=1, size=fading_baseband.shape)
        elif model == 'rice':
            magnitude = rice.rvs(K, size=fading_baseband.shape)
        elif model == 'nakagami':
            magnitude = nakagami.rvs(m, size=fading_baseband.shape)
        else:
            raise ValueError("Unsupported fading model")
        
        phase = np.random.uniform(0, 2*np.pi, fading_baseband.shape)
        return magnitude * np.exp(1j*phase)
    
    def awgn(self, snr_db, signal_power, size):
        """
        生成AWGN噪声
        :param snr_db: 信噪比 (dB)
        :param signal_power: 信号功率
        :param size: 噪声样本数
        :return: AWGN噪声
        """
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear
        return np.sqrt(noise_power/2) * (np.random.randn(*size) + 1j*np.random.randn(*size))
    
    def channel_characteristics(self, fading_signal):
        """
        评估信道特性
        :param fading_signal: 衰落信号（时变复信号）
        :return: 时间相关函数, 频率相关函数, 功率谱密度, 频率轴
        """
        # 时间相关函数（使用自相关）
        r_tt = np.correlate(fading_signal, fading_signal, mode='full')
        r_tt = r_tt / np.max(r_tt)  # 归一化
        
        # 频率轴计算（修正为基于信号长度的正确频率范围）
        fs = 1 / (0.01 / len(fading_signal))  # 假设采样周期为0.01/len(fading_signal)
        f = np.linspace(-fs/2, fs/2, len(fading_signal))  # 对称频率轴
        
        # 频率相关函数（通过FFT计算幅度相干函数）
        H_f = np.fft.fftshift(np.fft.fft(fading_signal))  # 中心频率移到中间
        r_ff = np.abs(np.correlate(H_f, H_f, mode='full')) / (np.abs(H_f)**2).sum()
        
        # 功率谱密度（带通信号PSD）
        psd = np.abs(H_f)**2 / len(fading_signal)
        
        return r_tt, r_ff, psd, f  # 返回频率轴f

if __name__ == "__main__":
    # 初始化信道模型
    channel = WirelessChannel()
    
    # 路径损耗计算
    d = np.logspace(2, 5, 1000)  # 100m到100km
    pl = channel.hata_path_loss(d)
    
    # 阴影衰落仿真
    shadow = channel.shadow_fading(size=1000)
    
    # 快衰落仿真（修正维度问题）
    fading_rayleigh = channel.jakes_fading(model='rayleigh')
    fading_rice = channel.jakes_fading(model='rice', K=10)
    fading_nakagami = channel.jakes_fading(model='nakagami', m=2)
    
    # AWGN噪声生成
    signal = np.ones(1000)
    snr_db = 20
    noise = channel.awgn(snr_db, np.var(signal), signal.shape)
    
    # 信道特性评估（解包包含频率轴）
    r_tt, r_ff, psd, f = channel.channel_characteristics(fading_rayleigh)
    
    # 绘制结果（所有中文标注）
    plt.figure(figsize=(15, 12))
    
    plt.subplot(321)
    plt.semilogx(d, pl)
    plt.title("路径损耗（Hata模型）")
    plt.xlabel("距离（米）")
    plt.ylabel("路径损耗（dB）")
    
    plt.subplot(322)
    plt.plot(shadow)
    plt.title("阴影衰落")
    plt.xlabel("样本索引")
    plt.ylabel("衰落（dB）")
    
    plt.subplot(323)
    plt.plot(np.abs(fading_rayleigh), label='Rayleigh 瑞利衰落')
    plt.plot(np.abs(fading_rice), label='Rice 莱斯衰落 (K=10)')
    plt.plot(np.abs(fading_nakagami), label='Nakagami 衰落 (m=2)')
    plt.title("快衰落包络")
    plt.xlabel("样本索引")
    plt.ylabel("幅度")
    plt.legend()
    
    plt.subplot(324)
    plt.plot(np.real(signal + noise), label='加性高斯白噪声信号')
    plt.title("AWGN 加性高斯白噪声信道")
    plt.xlabel("样本索引")
    plt.ylabel("幅度")
    
    plt.subplot(325)
    delay = np.linspace(-len(r_tt)//2, len(r_tt)//2, len(r_tt)) * (0.01/len(fading_rayleigh))  # 时间延迟轴
    plt.plot(delay, np.abs(r_tt))
    plt.title("时间相关函数")
    plt.xlabel("时间延迟（秒）")
    plt.ylabel("相关性")
    
    plt.subplot(326)
    plt.plot(f, psd)
    plt.title("功率谱密度（PSD）")
    plt.xlabel("频率（Hz）")
    plt.ylabel("功率谱密度（V²/Hz）")
    
    plt.tight_layout()
    plt.show()