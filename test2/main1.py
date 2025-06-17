import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import upfirdn
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

def rrc_filter(beta, sps, num_taps):
    """设计根升余弦滤波器"""
    T = 1
    t = np.arange(-num_taps//2, num_taps//2 + 1) / sps
    h = np.zeros_like(t)
    for i in range(len(t)):
        if t[i] == 0.0:
            h[i] = 1.0 - beta + (4 * beta / np.pi)
        elif abs(t[i]) == T / (4 * beta):
            h[i] = (beta / np.sqrt(2)) * (
                ((1 + 2/np.pi) * (np.sin(np.pi/(4*beta)))) +
                ((1 - 2/np.pi) * (np.cos(np.pi/(4*beta))))
            )
        else:
            h[i] = (np.sin(np.pi * t[i] * (1 - beta) / T) +
                    4 * beta * t[i] * np.cos(np.pi * t[i] * (1 + beta) / T) / T) / \
                   (np.pi * t[i] * (1 - (4 * beta * t[i] / T) ** 2) / T)
    h = h / np.sqrt(np.sum(h**2))
    return h

def plot_eye(signal, sps, num_symbols=5, offset=0, title="眼图"):
    """绘制眼图"""
    plt.figure(figsize=(8,4))
    for i in range(num_symbols * 20):
        start = offset + i * sps
        end = start + 2 * sps
        if end < len(signal):
            plt.plot(np.real(signal[start:end]), color='b', alpha=0.2)
    plt.title(title)
    plt.xlabel("采样点")
    plt.ylabel("幅度")
    plt.grid(True)
    plt.show()

def simulate_rrc_bpsk():
    """RSC滤波器和BPSK调制仿真"""
    # 参数设置
    N = 1000  # 总比特数
    rolloff = 0.25  # 滚降系数
    span = 8  # 符号周期数
    sps = 8  # 每个符号的采样点数
    num_taps = span * sps
    
    # 生成RRC滤波器
    rrc = rrc_filter(rolloff, sps, num_taps)
    
    # 生成随机比特
    bits = np.random.randint(0, 2, N)
    symbols = 2*bits - 1
    upsampled = upfirdn([1], symbols, sps)
    tx_signal = np.convolve(upsampled, rrc, mode='same')
    
    # 信道模型（平坦瑞利衰落+AWGN）
    h = (np.random.randn() + 1j*np.random.randn()) / np.sqrt(2)
    snr_db = 10
    snr_linear = 10**(snr_db/10)
    signal_power = np.mean(np.abs(tx_signal)**2)
    noise_power = signal_power / snr_linear
    noise = np.sqrt(noise_power/2) * (np.random.randn(len(tx_signal)) + 1j*np.random.randn(len(tx_signal)))
    rx_signal = h * tx_signal + noise
    
    # 接收端匹配滤波
    rx_filtered = np.convolve(rx_signal, rrc, mode='same')
    delay = int((len(rrc)-1)/2)
    sample_points = np.arange(delay, len(rx_filtered), sps)
    rx_samples = rx_filtered[sample_points]
    
    # 信道均衡
    rx_samples_eq = rx_samples / h
    detected_bits = (np.real(rx_samples_eq) > 0).astype(int)
    
    # 计算BER
    bits_rx = detected_bits[:N]
    ber = np.sum(bits != bits_rx) / N
    
    # 绘制结果
    plt.figure(figsize=(14,10))
    plt.subplot(3,1,1)
    plt.title("发送端RRC滤波后信号（实部，前200点）")
    plt.plot(np.real(tx_signal[:200]))
    plt.xlabel("采样点")
    plt.ylabel("幅度")
    plt.grid(True)
    
    plt.subplot(3,1,2)
    plt.title("接收端RRC滤波后信号（实部，前200点）")
    plt.plot(np.real(rx_filtered[:200]))
    plt.xlabel("采样点")
    plt.ylabel("幅度")
    plt.grid(True)
    
    plt.subplot(3,1,3)
    plt.title("接收端判决前采样点星座图")
    plt.scatter(np.real(rx_samples_eq), np.imag(rx_samples_eq), color='b', s=10)
    plt.xlabel("实部")
    plt.ylabel("虚部")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('stage2_rrc_bpsk.png')
    plt.close()
    
    print(f"RRC-BPSK仿真完成，BER = {ber:.4e}")
    return ber

def simulate_ber_vs_snr():
    """BER随SNR曲线仿真"""
    # 参数设置
    N = 1000  # 总比特数
    rolloff = 0.25  # 滚降系数
    span = 8  # 符号周期数
    sps = 8  # 每个符号的采样点数
    num_taps = span * sps
    snr_db_range = np.arange(0, 16, 2)
    num_trials = 20
    
    # 生成RRC滤波器
    rrc = rrc_filter(rolloff, sps, num_taps)
    
    ber_list = []
    for snr_db in snr_db_range:
        ber_sum = 0
        for _ in range(num_trials):
            # 生成随机比特
            bits = np.random.randint(0, 2, N)
            symbols = 2*bits - 1
            upsampled = upfirdn([1], symbols, sps)
            tx_signal = np.convolve(upsampled, rrc, mode='same')
            
            # 信道模型
            h = (np.random.randn() + 1j*np.random.randn()) / np.sqrt(2)
            snr_linear = 10**(snr_db/10)
            signal_power = np.mean(np.abs(tx_signal)**2)
            noise_power = signal_power / snr_linear
            noise = np.sqrt(noise_power/2) * (np.random.randn(len(tx_signal)) + 1j*np.random.randn(len(tx_signal)))
            rx_signal = h * tx_signal + noise
            
            # 接收端处理
            rx_filtered = np.convolve(rx_signal, rrc, mode='same')
            delay = int((len(rrc)-1)/2)
            sample_points = np.arange(delay, len(rx_filtered), sps)
            rx_samples = rx_filtered[sample_points]
            rx_samples_eq = rx_samples / h
            detected_bits = (np.real(rx_samples_eq) > 0).astype(int)
            
            # 计算BER
            bits_rx = detected_bits[:N]
            num_errors = np.sum(bits != bits_rx)
            ber_sum += num_errors / N
            
        ber_list.append(ber_sum / num_trials)
    
    # 绘制BER曲线
    plt.figure()
    plt.semilogy(snr_db_range, ber_list, 'o-', label='仿真BER')
    plt.xlabel('信噪比 SNR (dB)')
    plt.ylabel('比特误码率 BER')
    plt.title('BER随SNR变化曲线')
    plt.grid(True, which='both')
    plt.legend()
    plt.savefig('stage2_ber_vs_snr.png')
    plt.close()
    
    print("BER vs SNR仿真完成")

if __name__ == "__main__":
    simulate_rrc_bpsk()
    simulate_ber_vs_snr()