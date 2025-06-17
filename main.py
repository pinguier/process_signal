import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rayleigh, rice, nakagami
from scipy.signal import coherence
import matplotlib
from scipy.linalg import toeplitz
from numpy.linalg import inv
from tqdm import tqdm
from typing import Any, List, Tuple, Union, Optional

# 设置中文字体和支持负号
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

class WirelessChannel:
    def __init__(self, fc: float = 900e6, hb: float = 50, hm: float = 1.5, d0: float = 100, area: str = 'urban') -> None:
        """初始化无线信道参数"""
        self.fc = fc
        self.hb = hb
        self.hm = hm
        self.d0 = d0
        self.area = area.lower()
        self.c = 3e8  # 光速
        
    def hata_path_loss(self, d: float) -> float:
        """Hata模型路径损耗计算"""
        if self.area == 'urban':
            a_hm = (1.1 * np.log10(self.fc) - 0.7) * self.hm - (1.56 * np.log10(self.fc) - 0.8)
        elif self.area == 'suburban':
            a_hm = 0
        else:  # rural
            a_hm = 0
        
        L = 46.3 + 33.9 * np.log10(self.fc) - 13.82 * np.log10(self.hb) - a_hm + (44.9 - 6.55 * np.log10(self.hb)) * np.log10(d/self.d0)
        return L
    
    def shadow_fading(self, sigma: float = 8, size: int = 1) -> np.ndarray:
        """阴影衰落生成（对数正态分布）"""
        return np.random.normal(0, sigma, size)
    
    def jakes_fading(self, N: int = 100, theta: Optional[np.ndarray] = None, 
                     fd: float = 100, model: str = 'rayleigh', K: float = 10, m: float = 2) -> np.ndarray:
        """Jakes模型快衰落生成"""
        if theta is None:
            theta = 2 * np.pi * np.random.rand(N)
        
        t = np.linspace(0, 0.01, 1000)
        theta = theta[:, np.newaxis]
        t = t[np.newaxis, :]
        
        phase = 2 * np.pi * fd * t * np.cos(theta) + np.random.uniform(0, 2*np.pi, (N, 1))
        
        alpha = np.sqrt(2/N) * np.cos(phase)
        beta = np.sqrt(2/N) * np.sin(phase)
        fading_baseband = np.sum(alpha + 1j*beta, axis=0)
        
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
    
    def awgn(self, snr_db: float, signal_power: float, size: Tuple[int, ...]) -> np.ndarray:
        """生成AWGN噪声"""
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear
        return np.sqrt(noise_power/2) * (np.random.randn(*size) + 1j*np.random.randn(*size))

class Modulator:
    @staticmethod
    def bpsk_modulate(bits):
        """BPSK调制"""
        return 2 * bits - 1
    
    @staticmethod
    def qpsk_modulate(bits):
        """QPSK调制"""
        bits = bits.reshape((-1, 2))
        return (2*bits[:,0] - 1 + 1j*(2*bits[:,1] - 1)) / np.sqrt(2)
    
    @staticmethod
    def qam16_modulate(bits):
        """16QAM调制"""
        bits = bits.reshape((-1, 4))
        real = (2*bits[:,0] - 1) * (2 + 2*bits[:,1])
        imag = (2*bits[:,2] - 1) * (2 + 2*bits[:,3])
        return (real + 1j*imag) / np.sqrt(10)
    
    @staticmethod
    def bpsk_demodulate(symbols):
        """BPSK解调"""
        return (np.real(symbols) > 0).astype(int)
    
    @staticmethod
    def qpsk_demodulate(symbols):
        """QPSK解调"""
        bit1 = (np.real(symbols) > 0).astype(int)
        bit2 = (np.imag(symbols) > 0).astype(int)
        return np.hstack([bit1.reshape(-1,1), bit2.reshape(-1,1)]).flatten()
    
    @staticmethod
    def qam16_demodulate(symbols):
        """16QAM解调"""
        real = np.real(symbols)
        imag = np.imag(symbols)
        
        bits = np.zeros((len(symbols), 4))
        bits[:,0] = (real > 0).astype(int)
        bits[:,1] = (np.abs(real) > 2/np.sqrt(10)).astype(int)
        bits[:,2] = (imag > 0).astype(int)
        bits[:,3] = (np.abs(imag) > 2/np.sqrt(10)).astype(int)
        
        return bits.flatten()

class ChannelCoding:
    def __init__(self, constraint_length=3, code_rate=1/2):
        """初始化RSC编码器"""
        self.constraint_length = constraint_length
        self.code_rate = code_rate
        self.memory = np.zeros(constraint_length-1, dtype=np.int8)
        
    def encode(self, bits):
        """RSC编码"""
        # 确保输入比特是整数类型
        bits = bits.astype(np.int8)
        encoded_bits = []
        for bit in bits:
            # 更新移位寄存器
            self.memory = np.roll(self.memory, 1)
            self.memory[0] = bit
            
            # 生成编码比特
            encoded_bits.append(bit)  # 系统比特
            # 使用异或运算计算校验比特
            parity_bit = int(bit) ^ int(self.memory[0]) ^ int(self.memory[1])
            encoded_bits.append(parity_bit)
        
        return np.array(encoded_bits, dtype=np.int8)
    
    def viterbi_decode(self, received_bits):
        """Viterbi解码"""
        # 简化的Viterbi解码实现
        decoded_bits = []
        for i in range(0, len(received_bits), 2):
            if i+1 < len(received_bits):
                # 使用硬判决
                decoded_bits.append(received_bits[i])
        
        return np.array(decoded_bits, dtype=np.int8)

class Interleaver:
    def __init__(self, block_size):
        """初始化交织器"""
        self.block_size = block_size
        self.permutation = np.random.permutation(block_size)
        self.depermutation = np.argsort(self.permutation)
    
    def interleave(self, bits):
        """交织"""
        return bits[self.permutation]
    
    def deinterleave(self, bits):
        """解交织"""
        return bits[self.depermutation]

class Equalizer:
    @staticmethod
    def zf_equalize(y, H):
        """ZF均衡"""
        H_inv = np.linalg.pinv(H)
        return H_inv @ y
    
    @staticmethod
    def mmse_equalize(y, H, snr):
        """MMSE均衡"""
        Nt = H.shape[1]
        I = np.eye(Nt)
        H_H = H.conj().T @ H
        H_inv = np.linalg.inv(H_H + I/snr) @ H.conj().T
        return H_inv @ y

class MIMOSystem:
    def __init__(self, Nt, Nr):
        """初始化MIMO系统"""
        self.Nt = Nt
        self.Nr = Nr
    
    def generate_channel(self, correlation_t=None, correlation_r=None):
        """生成MIMO信道矩阵"""
        H_iid = (np.random.randn(self.Nr, self.Nt) + 1j * np.random.randn(self.Nr, self.Nt)) / np.sqrt(2)
        
        if correlation_t is not None and correlation_r is not None:
            R_r_sqrt = np.linalg.cholesky(correlation_r)
            R_t_sqrt = np.linalg.cholesky(correlation_t)
            H = R_r_sqrt @ H_iid @ R_t_sqrt.conj().T
        else:
            H = H_iid
            
        return H
    
    def stbc_encode(self, symbols):
        """STBC编码 (Alamouti码)"""
        assert len(symbols) % 2 == 0, "符号数量必须是偶数"
        symbols = symbols.reshape(-1, 2)
        num_pairs = len(symbols)
        
        encoded_symbols = np.zeros((2, 2*num_pairs), dtype=complex)
        
        for i in range(num_pairs):
            encoded_symbols[0, 2*i] = symbols[i, 0]
            encoded_symbols[1, 2*i] = symbols[i, 1]
            encoded_symbols[0, 2*i+1] = -symbols[i, 1].conj()
            encoded_symbols[1, 2*i+1] = symbols[i, 0].conj()
        
        return encoded_symbols

class OFDMSystem:
    def __init__(self, N, cp_length):
        """初始化OFDM系统"""
        self.N = N
        self.cp_length = cp_length
    
    def modulate(self, symbols):
        """OFDM调制"""
        assert len(symbols) == self.N, "符号数量必须等于FFT大小"
        time_signal = np.fft.ifft(symbols) * np.sqrt(self.N)
        cp = time_signal[-self.cp_length:]
        return np.concatenate([cp, time_signal])
    
    def demodulate(self, signal):
        """OFDM解调"""
        time_signal = signal[self.cp_length:]
        return np.fft.fft(time_signal) / np.sqrt(self.N)

def simulate_complete_system():
    """模拟完整的无线通信系统"""
    # 系统参数
    Nt = 4  # 发射天线数
    Nr = 4  # 接收天线数
    N = 128  # OFDM子载波数
    cp_length = 16  # 循环前缀长度
    num_bits = 9984  # 总比特数 (确保是OFDM符号大小的整数倍)
    snr_db_range = np.arange(0, 21, 2)  # SNR范围
    
    # 初始化各个模块
    channel = WirelessChannel()
    modulator = Modulator()
    channel_coding = ChannelCoding()
    interleaver = Interleaver(num_bits)  # 使用正确的比特数初始化交织器
    mimo_system = MIMOSystem(Nt, Nr)
    ofdm_system = OFDMSystem(N, cp_length)
    
    # 存储性能结果
    ber_results = np.zeros(len(snr_db_range))
    
    for snr_idx, snr_db in enumerate(snr_db_range):
        print(f"Processing SNR = {snr_db} dB")
        
        # 生成随机比特
        bits = np.random.randint(0, 2, num_bits)
        
        # 信道编码
        encoded_bits = channel_coding.encode(bits)
        
        # 交织
        interleaved_bits = interleaver.interleave(encoded_bits)
        
        # QPSK调制
        symbols = modulator.qpsk_modulate(interleaved_bits)
        
        # 分成多个OFDM符号
        symbols_per_ofdm = N
        num_ofdm_symbols = len(symbols) // symbols_per_ofdm
        
        received_symbols = []
        for i in range(num_ofdm_symbols):
            # 提取当前OFDM符号
            current_symbols = symbols[i*symbols_per_ofdm:(i+1)*symbols_per_ofdm]
            
            # OFDM调制
            ofdm_signal = ofdm_system.modulate(current_symbols)
            
            # 生成MIMO信道
            H = mimo_system.generate_channel()
            
            # 通过信道传输
            received_signal = np.zeros((Nr, len(ofdm_signal)), dtype=complex)
            for tx_ant in range(Nt):
                for rx_ant in range(Nr):
                    received_signal[rx_ant] += H[rx_ant, tx_ant] * ofdm_signal
            
            # 添加噪声
            snr = 10 ** (snr_db / 10)
            noise = channel.awgn(snr_db, 1, received_signal.shape)
            received_signal += noise
            
            # OFDM解调
            received_ofdm = np.zeros((Nr, N), dtype=complex)
            for rx_ant in range(Nr):
                received_ofdm[rx_ant] = ofdm_system.demodulate(received_signal[rx_ant])
            
            # MMSE检测
            detected_symbols = np.zeros(N, dtype=complex)
            for k in range(N):
                y_k = received_ofdm[:, k]
                H_k = H
                detected_symbols[k] = Equalizer.mmse_equalize(y_k, H_k, snr)[0]
            
            received_symbols.extend(detected_symbols)
        
        # QPSK解调
        received_bits = modulator.qpsk_demodulate(np.array(received_symbols))
        
        # 解交织
        deinterleaved_bits = interleaver.deinterleave(received_bits)
        
        # 信道解码
        decoded_bits = channel_coding.viterbi_decode(deinterleaved_bits)
        
        # 计算BER
        ber = np.sum(decoded_bits != bits) / len(bits)
        ber_results[snr_idx] = ber
    
    # 绘制BER曲线
    plt.figure(figsize=(10, 6))
    plt.semilogy(snr_db_range, ber_results, 'o-')
    plt.xlabel('SNR (dB)')
    plt.ylabel('BER')
    plt.title('完整无线通信系统的BER性能')
    plt.grid(True)
    plt.savefig('complete_system_ber.png')
    plt.close()
    
    print("仿真完成！结果已保存为complete_system_ber.png")

if __name__ == "__main__":
    print("开始完整无线通信系统仿真...")
    simulate_complete_system() 