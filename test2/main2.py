import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

def simulate_modulation_comparison():
    """BPSK/QPSK/16QAM调制比较仿真，包括基本性能和抗干扰能力评估"""
    # 仿真参数
    N = int(1e5)  # 每个SNR点仿真比特数
    SNR_dB = np.arange(0, 31, 2)  # SNR范围：0-30dB，步长2dB
    
    # 1. 基本性能评估
    print("正在进行基本性能评估...")
    ber_bpsk = []
    ber_qpsk = []
    ber_16qam = []
    
    for snr in SNR_dB:
        ber_bpsk.append(ber_bpsk_rayleigh(snr, N))
        ber_qpsk.append(ber_qpsk_rayleigh(snr, N))
        ber_16qam.append(ber_16qam_rayleigh(snr, N))
    
    # 绘制BER曲线
    plt.figure(figsize=(10, 6))
    plt.semilogy(SNR_dB, ber_bpsk, 'o-', label='BPSK')
    plt.semilogy(SNR_dB, ber_qpsk, 's-', label='QPSK')
    plt.semilogy(SNR_dB, ber_16qam, '^-', label='16QAM')
    plt.xlabel('信噪比 (dB)')
    plt.ylabel('误码率 (BER)')
    plt.title('平坦瑞利衰落信道下 BPSK/QPSK/16QAM 的误码率')
    plt.grid(True, which='both')
    plt.legend()
    plt.tight_layout()
    plt.savefig('stage2_modulation_comparison.png')
    plt.close()
    
    # 绘制频谱效率柱状图
    plt.figure(figsize=(8, 6))
    modulation_schemes = ['BPSK', 'QPSK', '16QAM']
    spectral_efficiency = [1, 2, 4]  # bit/s/Hz
    plt.bar(modulation_schemes, spectral_efficiency, color=['b', 'orange', 'g'])
    plt.ylabel('频谱效率 (bit/s/Hz)')
    plt.title('不同调制方式的频谱效率')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig('stage2_spectral_efficiency.png')
    plt.close()
    
    # 2. 抗干扰能力评估
    print("正在进行抗干扰能力评估...")
    interference_ratios = np.linspace(0, 1, 8)  # 干扰功率比范围：0-1，8个点
    snr_db = 10  # 固定SNR为10dB
    ber_bpsk_int = []
    ber_qpsk_int = []
    ber_16qam_int = []
    
    for ratio in interference_ratios:
        ber_bpsk_int.append(ber_bpsk_rayleigh_with_interference(snr_db, N, ratio))
        ber_qpsk_int.append(ber_qpsk_rayleigh_with_interference(snr_db, N, ratio))
        ber_16qam_int.append(ber_16qam_rayleigh_with_interference(snr_db, N, ratio))
    
    # 绘制抗干扰性能曲线
    plt.figure(figsize=(10, 6))
    plt.semilogy(interference_ratios, ber_bpsk_int, 'o-', label='BPSK')
    plt.semilogy(interference_ratios, ber_qpsk_int, 's-', label='QPSK')
    plt.semilogy(interference_ratios, ber_16qam_int, '^-', label='16QAM')
    plt.xlabel('干扰功率/信号功率')
    plt.ylabel('误码率 (BER)')
    plt.title('不同干扰功率下的抗干扰能力评估 (SNR=10dB)')
    plt.grid(True, which='both')
    plt.legend()
    plt.tight_layout()
    plt.savefig('stage2_interference_analysis.png')
    plt.close()
    
    print("调制比较仿真完成")

def ber_bpsk_rayleigh(snr_db, N):
    """BPSK在瑞利衰落信道下的BER计算
    
    Args:
        snr_db: 信噪比(dB)
        N: 仿真比特数
    
    Returns:
        float: 误码率
    """
    snr = 10**(snr_db/10)
    bits = np.random.randint(0, 2, N)
    symbols = 2*bits - 1
    h = (np.random.randn(N) + 1j*np.random.randn(N)) / np.sqrt(2)
    n = (np.random.randn(N) + 1j*np.random.randn(N)) / np.sqrt(2*snr)
    r = h * symbols + n
    r_eq = r / h
    bits_hat = (r_eq.real > 0).astype(int)
    return np.mean(bits != bits_hat)

def ber_qpsk_rayleigh(snr_db, N):
    """QPSK在瑞利衰落信道下的BER计算
    
    Args:
        snr_db: 信噪比(dB)
        N: 仿真比特数
    
    Returns:
        float: 误码率
    """
    snr = 10**(snr_db/10)
    bits = np.random.randint(0, 2, (N, 2))
    symbols = (2*bits[:,0]-1) + 1j*(2*bits[:,1]-1)
    symbols /= np.sqrt(2)
    h = (np.random.randn(N) + 1j*np.random.randn(N)) / np.sqrt(2)
    n = (np.random.randn(N) + 1j*np.random.randn(N)) / np.sqrt(2*snr)
    r = h * symbols + n
    r_eq = r / h
    bits_hat = np.zeros_like(bits)
    bits_hat[:,0] = (r_eq.real > 0).astype(int)
    bits_hat[:,1] = (r_eq.imag > 0).astype(int)
    return np.mean(bits != bits_hat)

def ber_16qam_rayleigh(snr_db, N):
    """16QAM在瑞利衰落信道下的BER计算
    
    Args:
        snr_db: 信噪比(dB)
        N: 仿真比特数
    
    Returns:
        float: 误码率
    """
    snr = 10**(snr_db/10)
    bits = np.random.randint(0, 2, (N, 4))
    mapping = np.array([[-3,-1,3,1], [-3,-1,3,1]])
    I = mapping[0, 2*bits[:,0]+bits[:,2]]
    Q = mapping[1, 2*bits[:,1]+bits[:,3]]
    symbols = (I + 1j*Q) / np.sqrt(10)
    h = (np.random.randn(N) + 1j*np.random.randn(N)) / np.sqrt(2)
    n = (np.random.randn(N) + 1j*np.random.randn(N)) / np.sqrt(2*snr)
    r = h * symbols + n
    r_eq = r / h
    I_hat = np.clip(np.round(r_eq.real * np.sqrt(10)), -3, 3)
    Q_hat = np.clip(np.round(r_eq.imag * np.sqrt(10)), -3, 3)
    I_hat = 2*((I_hat > 0).astype(int)) + (np.abs(I_hat)==1).astype(int)
    Q_hat = 2*((Q_hat > 0).astype(int)) + (np.abs(Q_hat)==1).astype(int)
    bits_hat = np.zeros_like(bits)
    bits_hat[:,0] = (I_hat >= 2).astype(int)
    bits_hat[:,2] = (I_hat % 2).astype(int)
    bits_hat[:,1] = (Q_hat >= 2).astype(int)
    bits_hat[:,3] = (Q_hat % 2).astype(int)
    return np.mean(bits != bits_hat)

def ber_bpsk_rayleigh_with_interference(snr_db, N, interference_power_ratio=0.5):
    """BPSK在瑞利衰落信道下带干扰的BER计算
    
    Args:
        snr_db: 信噪比(dB)
        N: 仿真比特数
        interference_power_ratio: 干扰功率与信号功率之比
    
    Returns:
        float: 误码率
    """
    snr = 10**(snr_db/10)
    bits = np.random.randint(0, 2, N)
    symbols = 2*bits - 1
    h = (np.random.randn(N) + 1j*np.random.randn(N)) / np.sqrt(2)
    n = (np.random.randn(N) + 1j*np.random.randn(N)) / np.sqrt(2*snr)
    interference = np.sqrt(interference_power_ratio) * (
        np.random.randn(N) + 1j*np.random.randn(N)
    ) / np.sqrt(2)
    r = h * symbols + n + interference
    r_eq = r / h
    bits_hat = (r_eq.real > 0).astype(int)
    return np.mean(bits != bits_hat)

def ber_qpsk_rayleigh_with_interference(snr_db, N, interference_power_ratio=0.5):
    """QPSK在瑞利衰落信道下带干扰的BER计算
    
    Args:
        snr_db: 信噪比(dB)
        N: 仿真比特数
        interference_power_ratio: 干扰功率与信号功率之比
    
    Returns:
        float: 误码率
    """
    snr = 10**(snr_db/10)
    bits = np.random.randint(0, 2, (N, 2))
    symbols = (2*bits[:,0]-1) + 1j*(2*bits[:,1]-1)
    symbols /= np.sqrt(2)
    h = (np.random.randn(N) + 1j*np.random.randn(N)) / np.sqrt(2)
    n = (np.random.randn(N) + 1j*np.random.randn(N)) / np.sqrt(2*snr)
    interference = np.sqrt(interference_power_ratio) * (
        np.random.randn(N) + 1j*np.random.randn(N)
    ) / np.sqrt(2)
    r = h * symbols + n + interference
    r_eq = r / h
    bits_hat = np.zeros_like(bits)
    bits_hat[:,0] = (r_eq.real > 0).astype(int)
    bits_hat[:,1] = (r_eq.imag > 0).astype(int)
    return np.mean(bits != bits_hat)

def ber_16qam_rayleigh_with_interference(snr_db, N, interference_power_ratio=0.5):
    """16QAM在瑞利衰落信道下带干扰的BER计算
    
    Args:
        snr_db: 信噪比(dB)
        N: 仿真比特数
        interference_power_ratio: 干扰功率与信号功率之比
    
    Returns:
        float: 误码率
    """
    snr = 10**(snr_db/10)
    bits = np.random.randint(0, 2, (N, 4))
    mapping = np.array([[-3,-1,3,1], [-3,-1,3,1]])
    I = mapping[0, 2*bits[:,0]+bits[:,2]]
    Q = mapping[1, 2*bits[:,1]+bits[:,3]]
    symbols = (I + 1j*Q) / np.sqrt(10)
    h = (np.random.randn(N) + 1j*np.random.randn(N)) / np.sqrt(2)
    n = (np.random.randn(N) + 1j*np.random.randn(N)) / np.sqrt(2*snr)
    interference = np.sqrt(interference_power_ratio) * (
        np.random.randn(N) + 1j*np.random.randn(N)
    ) / np.sqrt(2)
    r = h * symbols + n + interference
    r_eq = r / h
    I_hat = np.clip(np.round(r_eq.real * np.sqrt(10)), -3, 3)
    Q_hat = np.clip(np.round(r_eq.imag * np.sqrt(10)), -3, 3)
    I_hat = 2*((I_hat > 0).astype(int)) + (np.abs(I_hat)==1).astype(int)
    Q_hat = 2*((Q_hat > 0).astype(int)) + (np.abs(Q_hat)==1).astype(int)
    bits_hat = np.zeros_like(bits)
    bits_hat[:,0] = (I_hat >= 2).astype(int)
    bits_hat[:,2] = (I_hat % 2).astype(int)
    bits_hat[:,1] = (Q_hat >= 2).astype(int)
    bits_hat[:,3] = (Q_hat % 2).astype(int)
    return np.mean(bits != bits_hat)

if __name__ == "__main__":
    simulate_modulation_comparison()