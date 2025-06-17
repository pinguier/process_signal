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

# 多径信道抽头（假设 3 条路径，瑞利衰落）
channel_taps = np.array([0.9, 0.5, 0.3])
L = len(channel_taps)

# ----------------------
# 误码统计
# ----------------------
errors_ZF = np.zeros(len(SNR_dBs))
errors_MMSE = np.zeros(len(SNR_dBs))

# 计算总比特数用于BER计算
total_bits = num_iteration * num_symbol * Ntx * b

# ----------------------
# 主仿真循环
# ----------------------
for idx, SNR_dB in enumerate(SNR_dBs):
    SNR_linear = 10 ** (SNR_dB / 10)          # 转换为线性值
    noise_var = 1 / SNR_linear                # 噪声方差
    
    for iter in range(num_iteration):
        # 1. 随机生成比特
        tx_bits = np.random.randint(0, M, (Ntx, N))
        x_mod = 2 * tx_bits - 1                     # BPSK 调制
        
        # 2. OFDM IFFT 变换 (频域到时域)
        tx_symbols = np.fft.ifft(x_mod, axis=1) * np.sqrt(N)  # 加入归一化因子
        
        # 3. 添加循环前缀（CP）
        tx_cp = np.concatenate([tx_symbols[:, -cp_len:], tx_symbols], axis=1)
        
        # 4. 创建MIMO频域信道矩阵 (Nrx×Ntx) 维度的矩阵，每个元素是信道系数
        H_freq = np.zeros((N, Nrx, Ntx), dtype=complex)
        
        for k in range(N):
            H_freq[k] = np.sqrt(0.5) * (np.random.randn(Nrx, Ntx) + 1j * np.random.randn(Nrx, Ntx))
        
        # 5. 发送信号经过信道
        rx_cp = np.zeros((Nrx, tx_cp.shape[1]), dtype=complex)
        
        # 简化处理：直接在时域进行信道卷积（实际上应该是对每个发射天线单独做卷积）
        for i in range(Ntx):
            for j in range(Nrx):
                # 对每对发射接收天线，我们考虑多径效应
                h_path = np.sqrt(0.5) * (np.random.randn(L) + 1j * np.random.randn(L)) * channel_taps
                rx_cp[j] += np.convolve(tx_cp[i], h_path, mode='same')
        
        # 6. 添加噪声
        noise = np.sqrt(noise_var/2) * (np.random.randn(*rx_cp.shape) + 1j * np.random.randn(*rx_cp.shape))
        rx_cp += noise
        
        # 7. 去除循环前缀
        rx_symbols = rx_cp[:, cp_len:]
        
        # 8. FFT 转换到频域
        rx_freq = np.fft.fft(rx_symbols, axis=1) / np.sqrt(N)  # 对应的归一化
        
        # 9. 对每个子载波独立进行 ZF 和 MMSE 均衡
        y_ZF = np.zeros((Ntx, N), dtype=complex)
        y_MMSE = np.zeros((Ntx, N), dtype=complex)
        
        for k in range(N):  # 遍历每一个子载波
            H_k = H_freq[k]  # 提取每个子载波对应的信道
            r_k = rx_freq[:, k]  # 该子载波的接收信号
            
            # Zero-Forcing (ZF) 均衡
            try:
                # 使用伪逆求解
                H_k_inv = np.linalg.pinv(H_k)
                y_ZF[:, k] = H_k_inv @ r_k
            except np.linalg.LinAlgError:
                y_ZF[:, k] = np.zeros(Ntx)
            
            # MMSE 均衡
            try:
                # MMSE 求解：(H^H*H + sigma^2*I)^-1 * H^H * r
                H_mmse_inv = np.linalg.inv(H_k.conj().T @ H_k + noise_var * np.eye(Ntx)) @ H_k.conj().T
                y_MMSE[:, k] = H_mmse_inv @ r_k
            except np.linalg.LinAlgError:
                y_MMSE[:, k] = np.zeros(Ntx)
        
        # 10. BPSK 解调
        y_demod_ZF = (np.real(y_ZF) > 0).astype(int)
        y_demod_MMSE = (np.real(y_MMSE) > 0).astype(int)
        
        # 11. 统计误码
        errors_ZF[idx] += np.sum(y_demod_ZF != tx_bits)
        errors_MMSE[idx] += np.sum(y_demod_MMSE != tx_bits)

# ----------------------
# 计算误码率
# ----------------------
error_rate_ZF = errors_ZF / total_bits
error_rate_MMSE = errors_MMSE / total_bits

# ----------------------
# 绘制 BER 对比图 (中文标签)
# ----------------------
plt.figure(figsize=(10, 6))
plt.semilogy(SNR_dBs, error_rate_ZF, 'r-o', linewidth=2, markersize=8, label='ZF 均衡器')
plt.semilogy(SNR_dBs, error_rate_MMSE, 'b-s', linewidth=2, markersize=8, label='MMSE 均衡器')
plt.grid(True, which='both', linestyle='--', alpha=0.7)
plt.title('基于多径衰落信道的ZF均衡器与MMSE均衡器的比较', fontsize=16)
plt.xlabel('信噪比 (SNR, dB)', fontsize=14)
plt.ylabel('比特误码率 (BER)', fontsize=14)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig('stage3_ofdm_performance.png')
plt.close()

### 绘制 信号和噪声的时域波形图
plt.figure(figsize=(10, 6))
plt.plot(np.real(rx_cp[0]), label="接收信号 (实部)", linestyle='--')
plt.plot(np.real(tx_cp[0]), label="发送信号 (实部)", linestyle='-')
plt.xlabel('时间')
plt.ylabel('幅度')
plt.title('接收信号与发送信号时域波形')
plt.legend()
plt.grid(True)
plt.savefig('stage3_ofdm_time_domain.png')
plt.close()

## 绘制频域信号的功率谱图

plt.figure(figsize=(10, 6))
plt.psd(np.real(tx_cp[0]), NFFT=1024, Fs=1, label='发送信号')
plt.psd(np.real(rx_cp[0]), NFFT=1024, Fs=1, label='接收信号')
plt.xlabel('频率')
plt.ylabel('功率谱密度')
plt.title('发送信号与接收信号的功率谱')
plt.legend()
plt.grid(True)
plt.savefig('stage3_ofdm_frequency_domain.png')
plt.close()



plt.figure(figsize=(10, 6))
plt.scatter(np.real(y_ZF), np.imag(y_ZF), label="ZF 均衡后信号")
plt.scatter(np.real(x_mod), np.imag(x_mod), label="发送信号", alpha=0.5)
plt.title("ZF 均衡后的信号与原始信号对比")
plt.xlabel("实部")
plt.ylabel("虚部")
plt.legend()
plt.grid(True)
plt.savefig('stage3_ofdm_zf_signal.png')
plt.close()


plt.figure(figsize=(10, 6))
plt.scatter(np.real(y_MMSE), np.imag(y_MMSE), label="MMSE 均衡后信号")
plt.scatter(np.real(x_mod), np.imag(x_mod), label="发送信号", alpha=0.5)
plt.title("MMSE 均衡后的信号与原始信号对比")
plt.xlabel("实部")
plt.ylabel("虚部")
plt.legend()
plt.grid(True)
plt.savefig('stage3_ofdm_mmse_signal.png')
plt.close()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

def simulate_ofdm_performance():
    """OFDM系统性能分析仿真"""
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
    plt.savefig('stage3_ofdm_performance.png')
    plt.close()
    
    print("OFDM性能分析仿真完成")

if __name__ == "__main__":
    simulate_ofdm_performance()