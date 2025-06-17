import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# ----------------------
# 基本参数设置
# ----------------------
N = 64                  # OFDM 子载波数
cp_len = 16             # 循环前缀长度
num_symbols = 1000      # 每个SNR下发送的OFDM符号数量
SNR_dBs = np.arange(-10, 21, 2)  # SNR从-10到20 dB

# 多径信道抽头（归一化能量）
channel_taps = np.array([0.4, 0.3, 0.2])
channel_taps = channel_taps / np.sqrt(np.sum(np.abs(channel_taps)**2))  # 确保信道能量归一化
L = len(channel_taps)  # 信道长度（路径数）

# ----------------------
# 调制与解调函数（BPSK）
# ----------------------
def bpsk_mod(bits):
    """将比特映射为BPSK符号（0 -> -1, 1 -> 1）"""
    return 2 * bits - 1

def bpsk_demod(symbols):
    """BPSK解调，基于实部判断（>0 -> 1, <=0 -> 0）"""
    return (symbols.real > 0).astype(int)

# 添加/移除循环前缀
def add_cp(signal, cp_len):
    """添加循环前缀，复制信号末尾cp_len个样本"""
    return np.concatenate([signal[-cp_len:], signal])

def remove_cp(signal, cp_len):
    """移除循环前缀，从第cp_len个样本开始"""
    return signal[cp_len:]

# 多径信道建模
def apply_channel(signal, h):
    """应用多径信道，通过卷积模拟信道效应"""
    return np.convolve(signal, h)[:len(signal)]

# ZF均衡器函数
def zf_equalizer(rx_freq, H_freq):
    """ZF均衡器，消除信道效应，添加小常数避免除零"""
    eps = 1e-20
    H_zf = np.where(np.abs(H_freq) > eps, 1.0 / H_freq, 0)
    rx_zf = rx_freq * H_zf
    return rx_zf

# MMSE均衡器函数
def mmse_equalizer(rx_freq, H_freq, noise_var):
    """MMSE均衡器，考虑噪声方差，优化低SNR性能"""
    freq_noise_var = noise_var  # 正确使用时域噪声方差
    H_mmse = np.conj(H_freq) / (np.abs(H_freq)**2 + freq_noise_var)
    rx_mmse = rx_freq * H_mmse
    return rx_mmse

# ----------------------
# 主仿真开始
# ----------------------
np.random.seed(42)  # 设置随机种子，确保结果可重复
ber_zf = []
ber_mmse = []

for SNR_dB in SNR_dBs:
    bit_errors_zf = 0
    bit_errors_mmse = 0
    total_bits = 0
    
    for _ in range(num_symbols):
        # 1. 随机生成比特
        bits = np.random.randint(0, 2, N)
        symbols = bpsk_mod(bits)
        
        # 2. OFDM调制（IFFT + CP）
        tx_time = ifft(symbols, norm='ortho')
        tx_cp = add_cp(tx_time, cp_len)
        
        # 3. 多径瑞利衰落信道（随机复数抽头）
        h_time = channel_taps * (np.random.randn(L) + 1j * np.random.randn(L)) / np.sqrt(2)
        rx_time = apply_channel(tx_cp, h_time)
        
        # 假设单位信号功率（BPSK）
        tx_power = 1.0
        noise_var = tx_power * 10 ** (-SNR_dB / 10)
        
        # 4. 添加复高斯噪声
        noise = np.sqrt(noise_var / 2) * (np.random.randn(len(rx_time)) + 1j * np.random.randn(len(rx_time)))
        rx_time_noisy = rx_time + noise
        
        # 5. 接收端去除CP + FFT
        rx_no_cp = remove_cp(rx_time_noisy, cp_len)
        rx_freq = fft(rx_no_cp, norm='ortho')
        
        # 6. 完美信道估计
        h_padded = np.zeros(N, dtype=complex)
        h_padded[:len(h_time)] = h_time
        H_freq = fft(h_padded, norm='ortho')
        
        # 7. Zero-Forcing 均衡
        rx_zf = zf_equalizer(rx_freq, H_freq)
        bits_zf = bpsk_demod(rx_zf)
        bit_errors_zf += np.sum(bits != bits_zf)
        
        # 8. MMSE 均衡
        rx_mmse = mmse_equalizer(rx_freq, H_freq, noise_var)
        bits_mmse = bpsk_demod(rx_mmse)
        bit_errors_mmse += np.sum(bits != bits_mmse)
        
        total_bits += N
    
    # 9. 记录BER
    ber_zf.append(bit_errors_zf / total_bits if total_bits > 0 else 1.0)
    ber_mmse.append(bit_errors_mmse / total_bits if total_bits > 0 else 1.0)
    
    print(f"SNR = {SNR_dB} dB | ZF BER = {ber_zf[-1]:.6f} | MMSE BER = {ber_mmse[-1]:.6f} | 差异: {ber_zf[-1] - ber_mmse[-1]:.6f}")

# ----------------------
# 结果绘图
# ----------------------
plt.figure(figsize=(10, 6))
plt.semilogy(SNR_dBs, ber_zf, 'o-', linewidth=2, label='ZF Equalizer')
plt.semilogy(SNR_dBs, ber_mmse, 's-', linewidth=2, label='MMSE Equalizer', color='red')
plt.xlabel('SNR (dB)', fontsize=12)
plt.ylabel('Bit Error Rate (BER)', fontsize=12)
plt.title('BER Comparison: ZF vs MMSE under Frequency-Selective Channel', fontsize=14)
plt.grid(True, which='both')
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig('stage3_ofdm_basic.png')
plt.close()

# 打印差异数据
print("\nZF与MMSE的BER差异:")
for i, snr in enumerate(SNR_dBs):
    print(f"SNR = {snr} dB | 差异 = {ber_zf[i] - ber_mmse[i]:.6f}")

def simulate_ofdm_basic():
    """OFDM基本功能仿真"""
    # 仿真参数
    N = 64  # 子载波数
    CP = 16  # 循环前缀长度
    N_bits = 10000  # 仿真比特数
    SNR_dB = np.arange(-10, 21, 2)  # SNR范围：-10到20dB
    
    # 存储结果
    ber = []
    
    # 生成随机比特
    bits = np.random.randint(0, 2, N_bits)
    
    # BPSK调制
    symbols = 2*bits - 1
    
    # 对每个SNR进行仿真
    for snr in SNR_dB:
        # 添加噪声
        noise_power = 10**(-snr/10)
        noise = np.sqrt(noise_power/2) * (np.random.randn(N_bits) + 1j*np.random.randn(N_bits))
        
        # 通过信道
        h = (np.random.randn(N_bits) + 1j*np.random.randn(N_bits)) / np.sqrt(2)
        received = h * symbols + noise
        
        # 信道均衡
        equalized = received / h
        bits_hat = (equalized.real > 0).astype(int)
        ber.append(np.mean(bits != bits_hat))
        
        # 打印结果
        print(f"SNR = {snr:2d} dB, BER = {ber[-1]:.6f}")
    
    # 绘制性能曲线
    plt.figure(figsize=(10, 6))
    plt.semilogy(SNR_dB, ber, 'o-')
    plt.xlabel('信噪比 (dB)')
    plt.ylabel('误码率 (BER)')
    plt.title('OFDM系统基本性能')
    plt.grid(True, which='both')
    plt.tight_layout()
    plt.savefig('stage3_ofdm_basic.png')
    plt.close()
    
    print("OFDM基本功能仿真完成")

if __name__ == "__main__":
    simulate_ofdm_basic()