import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 设置中文字体和支持负号
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 显示中文
matplotlib.rcParams['axes.unicode_minus'] = False             # 支持负号

# ----------------------
# 参数初始化
# ----------------------
Nrx = int(8)                      # 接收天线数
Ntx = int(4)                      # 发送天线数
num_symbol = N = 64               # OFDM 子载波数
SNR_dBs = np.arange(0, 21, 2)     # SNR 范围（单位 dB）
b = 1                             # BPSK 的每符号比特数
M = 2 ** b                        # BPSK 调制
num_iteration = 100               # 仿真次数
cp_len = 16                       # 循环前缀长度

N = 64                        # 频域样本数（相当于子载波数）
num_iteration = 100
SNR_dBs = np.arange(0, 21, 2)
cp_len = 16
channel_taps = np.array([0.9, 0.5, 0.3])
L = len(channel_taps)

errors_ZF = np.zeros(len(SNR_dBs))
errors_MMSE = np.zeros(len(SNR_dBs))
total_bits = num_iteration * N

for idx, SNR_dB in enumerate(SNR_dBs):
    SNR_linear = 10 ** (SNR_dB / 10)
    noise_var = 1 / SNR_linear
    
    for _ in range(num_iteration):
        # 1. 生成比特 + BPSK 调制
        bits = np.random.randint(0, 2, N)
        symbols = 2 * bits - 1
        
        # 2. IFFT + 添加循环前缀
        time_signal = np.fft.ifft(symbols) * np.sqrt(N)
        tx_cp = np.concatenate([time_signal[-cp_len:], time_signal])
        
        # 3. 信道卷积 + 添加噪声
        h = (np.random.randn(L) + 1j * np.random.randn(L)) * channel_taps
        rx_cp = np.convolve(tx_cp, h, mode='same')
        noise = np.sqrt(noise_var / 2) * (np.random.randn(len(rx_cp)) + 1j * np.random.randn(len(rx_cp)))
        rx_cp += noise
        
        # 4. 去除 CP + FFT
        rx = rx_cp[cp_len:]
        rx_freq = np.fft.fft(rx) / np.sqrt(N)
        
        # 5. 构造频域信道响应（注意 padding）
        h_padded = np.pad(h, (0, N - L), 'constant')
        H_freq = np.fft.fft(h_padded)
        
        # 6. ZF & MMSE 均衡
        y_ZF = rx_freq / H_freq
        y_MMSE = np.conj(H_freq) * rx_freq / (np.abs(H_freq)**2 + noise_var)
        
        # 7. 解调
        demod_ZF = (np.real(y_ZF) > 0).astype(int)
        demod_MMSE = (np.real(y_MMSE) > 0).astype(int)
        
        # 8. 统计误码
        errors_ZF[idx] += np.sum(demod_ZF != bits)
        errors_MMSE[idx] += np.sum(demod_MMSE != bits)

# 9. 绘图
plt.figure()
plt.semilogy(SNR_dBs, errors_ZF / total_bits, 'r-o', label='ZF 均衡')
plt.semilogy(SNR_dBs, errors_MMSE / total_bits, 'b-s', label='MMSE 均衡')
plt.xlabel('SNR (dB)')
plt.ylabel('BER')
plt.title('频率选择性信道下 ZF 与 MMSE 均衡性能对比')
plt.grid(True, which='both')
plt.legend()
plt.savefig('stage3_zf_mmse_comparison.png')
plt.close()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

def simulate_ofdm_additional():
    """OFDM系统附加性能分析仿真"""
    # 仿真参数
    N = 64  # 子载波数
    CP = 16  # 循环前缀长度
    N_bits = 10000  # 仿真比特数
    SNR_dB = np.arange(-10, 21, 2)  # SNR范围：-10到20dB
    
    # 存储结果
    ber_no_cp = []
    ber_with_cp = []
    ber_ideal = []
    
    # 生成随机比特
    bits = np.random.randint(0, 2, N_bits)
    
    # BPSK调制
    symbols = 2*bits - 1
    
    # 对每个SNR进行仿真
    for snr in SNR_dB:
        # 添加噪声
        noise_power = 10**(-snr/10)
        
        # 1. 无CP的情况
        noise_no_cp = np.sqrt(noise_power/2) * (np.random.randn(N_bits) + 1j*np.random.randn(N_bits))
        h_no_cp = (np.random.randn(N_bits) + 1j*np.random.randn(N_bits)) / np.sqrt(2)
        received_no_cp = h_no_cp * symbols + noise_no_cp
        equalized_no_cp = received_no_cp / h_no_cp
        bits_no_cp = (equalized_no_cp.real > 0).astype(int)
        ber_no_cp.append(np.mean(bits != bits_no_cp))
        
        # 2. 有CP的情况
        # 添加CP
        symbols_with_cp = np.concatenate([symbols[-CP:], symbols])
        # 为带CP的信号生成相应长度的噪声和信道响应
        noise_with_cp = np.sqrt(noise_power/2) * (np.random.randn(N_bits + CP) + 1j*np.random.randn(N_bits + CP))
        h_with_cp = (np.random.randn(N_bits + CP) + 1j*np.random.randn(N_bits + CP)) / np.sqrt(2)
        received_with_cp = h_with_cp * symbols_with_cp + noise_with_cp
        # 移除CP
        received_with_cp = received_with_cp[CP:]
        h_with_cp = h_with_cp[CP:]  # 同样移除CP部分的信道响应
        equalized_with_cp = received_with_cp / h_with_cp
        bits_with_cp = (equalized_with_cp.real > 0).astype(int)
        ber_with_cp.append(np.mean(bits != bits_with_cp))
        
        # 3. 理想信道情况
        noise_ideal = np.sqrt(noise_power/2) * (np.random.randn(N_bits) + 1j*np.random.randn(N_bits))
        received_ideal = symbols + noise_ideal
        bits_ideal = (received_ideal.real > 0).astype(int)
        ber_ideal.append(np.mean(bits != bits_ideal))
        
        # 打印结果
        print(f"SNR = {snr:2d} dB, 无CP BER = {ber_no_cp[-1]:.6f}, 有CP BER = {ber_with_cp[-1]:.6f}, 理想信道 BER = {ber_ideal[-1]:.6f}")
    
    # 绘制性能曲线
    plt.figure(figsize=(10, 6))
    plt.semilogy(SNR_dB, ber_no_cp, 'o-', label='无CP')
    plt.semilogy(SNR_dB, ber_with_cp, 's-', label='有CP')
    plt.semilogy(SNR_dB, ber_ideal, '^-', label='理想信道')
    plt.xlabel('信噪比 (dB)')
    plt.ylabel('误码率 (BER)')
    plt.title('OFDM系统在不同配置下的性能比较')
    plt.grid(True, which='both')
    plt.legend()
    plt.tight_layout()
    plt.savefig('stage3_ofdm_additional.png')
    plt.close()
    
    # 分析CP的影响
    cp_improvement = np.array(ber_no_cp) / np.array(ber_with_cp)
    plt.figure(figsize=(10, 6))
    plt.plot(SNR_dB, cp_improvement, 'o-')
    plt.xlabel('信噪比 (dB)')
    plt.ylabel('性能改善倍数')
    plt.title('循环前缀带来的性能改善')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('stage3_cp_improvement.png')
    plt.close()
    
    print("OFDM附加性能分析仿真完成")

if __name__ == "__main__":
    simulate_ofdm_additional()