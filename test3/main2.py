import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

def simulate_ofdm_equalization():
    """OFDM信道均衡仿真"""
    # 仿真参数
    N = 64  # 子载波数
    CP = 16  # 循环前缀长度
    N_bits = 10000  # 仿真比特数
    SNR_dB = np.arange(-10, 21, 2)  # SNR范围：-10到20dB
    
    # 存储结果
    ber_zf = []
    ber_mmse = []
    
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
        
        # ZF均衡
        zf_equalized = received / h
        zf_bits = (zf_equalized.real > 0).astype(int)
        ber_zf.append(np.mean(bits != zf_bits))
        
        # MMSE均衡
        mmse_equalized = received * np.conj(h) / (np.abs(h)**2 + noise_power)
        mmse_bits = (mmse_equalized.real > 0).astype(int)
        ber_mmse.append(np.mean(bits != mmse_bits))
        
        # 打印结果
        print(f"SNR = {snr:2d} dB, ZF BER = {ber_zf[-1]:.6f}, MMSE BER = {ber_mmse[-1]:.6f}")
    
    # 绘制性能曲线
    plt.figure(figsize=(10, 6))
    plt.semilogy(SNR_dB, ber_zf, 'o-', label='ZF均衡')
    plt.semilogy(SNR_dB, ber_mmse, 's-', label='MMSE均衡')
    plt.xlabel('信噪比 (dB)')
    plt.ylabel('误码率 (BER)')
    plt.title('OFDM系统在不同均衡方式下的性能')
    plt.grid(True, which='both')
    plt.legend()
    plt.tight_layout()
    plt.savefig('stage3_equalization_comparison.png')
    plt.close()
    
    # 分析均衡器差异
    plt.figure(figsize=(10, 6))
    plt.plot(SNR_dB, np.array(ber_zf)/np.array(ber_mmse), 'o-')
    plt.xlabel('信噪比 (dB)')
    plt.ylabel('ZF/MMSE BER比值')
    plt.title('ZF与MMSE均衡器的性能差异')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('stage3_equalizer_difference.png')
    plt.close()
    
    print("OFDM信道均衡仿真完成")

def bpsk_mod(bits):
    """BPSK调制"""
    return 2 * bits - 1

def bpsk_demod(symbols):
    """BPSK解调"""
    return (symbols.real > 0).astype(int)

def add_cp(signal, cp_len):
    """添加循环前缀"""
    return np.concatenate([signal[-cp_len:], signal])

def remove_cp(signal, cp_len):
    """移除循环前缀"""
    return signal[cp_len:]

def apply_channel(signal, h):
    """应用多径信道"""
    return np.convolve(signal, h)[:len(signal)]

if __name__ == "__main__":
    simulate_ofdm_equalization()
