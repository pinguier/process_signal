import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz
from numpy.linalg import inv
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
# 设置中文字体和支持负号
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 显示中文
matplotlib.rcParams['axes.unicode_minus'] = False             # 支持负号

# 设置随机种子以确保结果可复现
np.random.seed(42)

# =====================================
# 第一部分：平坦瑞利衰落信道仿真
# =====================================

def generate_flat_rayleigh_channel(Nt, Nr, correlation_t=None, correlation_r=None):
    """
    生成平坦瑞利衰落信道矩阵，支持空间相关性
    
    参数:
    Nt: 发射天线数
    Nr: 接收天线数
    correlation_t: 发射端相关矩阵
    correlation_r: 接收端相关矩阵
    
    返回:
    H: 信道矩阵
    """
    # 生成独立同分布的复高斯信道
    H_iid = (np.random.randn(Nr, Nt) + 1j * np.random.randn(Nr, Nt)) / np.sqrt(2)
    
    # 如果提供了相关矩阵，则应用空间相关性
    if correlation_t is not None and correlation_r is not None:
        R_r_sqrt = np.linalg.cholesky(correlation_r)
        R_t_sqrt = np.linalg.cholesky(correlation_t)
        H = R_r_sqrt @ H_iid @ R_t_sqrt.conj().T
    else:
        H = H_iid
        
    return H

def analyze_spatial_correlation():
    """分析空间相关性对信道容量的影响"""
    Nt = 4
    Nr = 4
    snr_db_range = np.arange(0, 31, 5)
    rho_range = [0.0, 0.3, 0.7, 0.9]  # 相关系数
    
    plt.figure(figsize=(10, 6))
    
    for rho in rho_range:
        # 构建相关矩阵
        corr_vec = rho ** np.arange(Nt)
        correlation_matrix = toeplitz(corr_vec)
        
        capacities = []
        for snr_db in snr_db_range:
            snr = 10 ** (snr_db / 10)
            capacity_samples = []
            
            # 蒙特卡洛模拟
            for _ in range(1000):
                H = generate_flat_rayleigh_channel(Nt, Nr, correlation_matrix, correlation_matrix)
                # 计算信道容量
                I = np.eye(Nr)
                C = np.log2(np.linalg.det(I + snr/Nt * H @ H.conj().T)).real
                capacity_samples.append(C)
            
            capacities.append(np.mean(capacity_samples))
        
        plt.plot(snr_db_range, capacities, 'o-', label=f'ρ={rho}')
    
    plt.xlabel('SNR (dB)')
    plt.ylabel('Ergodic Capacity (bps/Hz)')
    plt.title('空间相关性对MIMO信道容量的影响')
    plt.legend()
    plt.grid(True)
    plt.savefig('spatial_correlation.png')
    plt.close()
    
    print("空间相关性分析完成，结果已保存为spatial_correlation.png")

def qpsk_modulate(bits):
    """QPSK调制"""
    # 将比特分成每两个一组
    bits = bits.reshape((-1, 2))
    # QPSK星座点：{1+1j, 1-1j, -1+1j, -1-1j} / sqrt(2)
    symbols = (2*bits[:,0] - 1 + 1j*(2*bits[:,1] - 1)) / np.sqrt(2)
    return symbols

def qpsk_demodulate(symbols):
    """QPSK解调"""
    # 实部大于0映射为1，否则为0
    bit1 = (np.real(symbols) > 0).astype(int)
    # 虚部大于0映射为1，否则为0
    bit2 = (np.imag(symbols) > 0).astype(int)
    # 合并比特
    bits = np.hstack([bit1.reshape(-1,1), bit2.reshape(-1,1)]).flatten()
    return bits

def stbc_encode(symbols):
    """
    STBC编码 (Alamouti码，2x2)
    
    参数:
    symbols: 输入符号，每两个一组
    
    返回:
    encoded_symbols: 编码后的符号，形状为[2, N/2]，表示两个时间时隙的发送符号
    """
    # 确保符号数量是偶数
    assert len(symbols) % 2 == 0, "符号数量必须是偶数"
    
    # 重组为每两个符号一组
    symbols = symbols.reshape(-1, 2)
    num_pairs = len(symbols)
    
    # 创建编码后的符号矩阵 [2, 2*num_pairs]
    encoded_symbols = np.zeros((2, 2*num_pairs), dtype=complex)
    
    # 应用Alamouti码
    for i in range(num_pairs):
        # 第一个时隙
        encoded_symbols[0, 2*i] = symbols[i, 0]      # 第一天线在t1时刻发送s1
        encoded_symbols[1, 2*i] = symbols[i, 1]      # 第二天线在t1时刻发送s2
        
        # 第二个时隙
        encoded_symbols[0, 2*i+1] = -symbols[i, 1].conj()  # 第一天线在t2时刻发送-s2*
        encoded_symbols[1, 2*i+1] = symbols[i, 0].conj()   # 第二天线在t2时刻发送s1*
    
    return encoded_symbols

def stbc_detect(y1, y2, h1, h2, snr):
    """
    STBC检测 (Alamouti码，2x2)
    
    参数:
    y1, y2: 两个时间时隙的接收信号
    h1, h2: 两个发射天线到接收天线的信道系数
    snr: 信噪比 (线性值)
    
    返回:
    detected_symbols: 检测到的符号
    """
    # STBC检测
    s1_hat = y1 * h1.conj() + y2 * h2
    
    # 对于Alamouti码，我们需要两个接收信号来检测两个符号
    detected_symbols = np.array([s1_hat]) / (np.abs(h1)**2 + np.abs(h2)**2 + 1/snr)
    
    return detected_symbols

def analyze_stbc_ber():
    """分析STBC编码的BER性能"""
    Nt = 2
    Nr = 2
    # 确保比特数是4的倍数
    num_bits = 100000
    assert num_bits % 4 == 0, "比特数必须是4的倍数"
    
    snr_db_range = np.arange(0, 21, 2)
    ber_stbc = np.zeros(len(snr_db_range))
    ber_no_stbc = np.zeros(len(snr_db_range))
    
    # 其余代码保持不变...
    
    for snr_idx, snr_db in enumerate(snr_db_range):
        snr = 10 ** (snr_db / 10)
        num_errors_stbc = 0
        num_errors_no_stbc = 0
        
        # 生成随机比特
        bits = np.random.randint(0, 2, num_bits)
        
        # QPSK调制
        symbols = qpsk_modulate(bits)
        
        # 对于STBC，符号需要成对处理
        stbc_symbols = stbc_encode(symbols)
        
        # 模拟传输
        for i in range(0, len(stbc_symbols[0])-1, 2):
            # 生成信道矩阵
            H = generate_flat_rayleigh_channel(Nt, Nr)
            
            # 发送两个符号，每个符号通过不同的天线
            s = np.array([stbc_symbols[0,i], stbc_symbols[1,i]])
            
            # 接收信号
            noise = (np.random.randn(Nr) + 1j * np.random.randn(Nr)) / np.sqrt(2*snr)
            y = H @ s + noise
            
            # STBC检测
            h1 = H[0, 0]  # 第一天线到接收天线的信道系数
            h2 = H[0, 1]  # 第二天线到接收天线的信道系数
            s_hat = stbc_detect(y[0], y[1], h1, h2, snr)
            
            # QPSK解调
            bits_hat = qpsk_demodulate(s_hat)
            
            # 计算错误数
            num_errors_stbc += np.sum(bits_hat != bits[i:i+2])
        
        # 没有STBC的情况 (简单MIMO)
        for i in range(0, len(symbols), Nt):
            # 生成信道矩阵
            H = generate_flat_rayleigh_channel(Nt, Nr)
            
            # 选择Nt个符号
            s = symbols[i:i+Nt]
            
            # 接收信号
            noise = (np.random.randn(Nr) + 1j * np.random.randn(Nr)) / np.sqrt(2*snr)
            y = H @ s + noise
            
            # ZF检测
            H_inv = np.linalg.pinv(H)
            s_hat = H_inv @ y
            
            # QPSK解调
            bits_hat = qpsk_demodulate(s_hat)
            
            # 计算错误数
            num_errors_no_stbc += np.sum(bits_hat != bits[i*2:(i+Nt)*2])
        
        # 计算BER
        ber_stbc[snr_idx] = num_errors_stbc / num_bits
        ber_no_stbc[snr_idx] = num_errors_no_stbc / num_bits
    
    # 绘制BER曲线
    plt.figure(figsize=(10, 6))
    plt.semilogy(snr_db_range, ber_stbc, 'o-', label='STBC (2x2)')
    plt.semilogy(snr_db_range, ber_no_stbc, 's-', label='No STBC (2x2)')
    plt.xlabel('SNR (dB)')
    plt.ylabel('BER')
    plt.title('STBC编码的BER性能')
    plt.grid(True)
    plt.legend()
    plt.savefig('stbc_ber.png')
    plt.close()
    
    print("STBC BER分析完成，结果已保存为stbc_ber.png")

def zf_detection(y, H):
    """ZF检测"""
    H_inv = np.linalg.pinv(H)
    s_hat = H_inv @ y
    return s_hat

def mmse_detection(y, H, snr):
    """MMSE检测"""
    Nt = H.shape[1]
    I = np.eye(Nt)
    H_H = H.conj().T @ H
    H_inv = np.linalg.inv(H_H + I/snr) @ H.conj().T
    s_hat = H_inv @ y
    return s_hat

def analyze_mimo_detection():
    """分析4x4 MIMO系统中ZF和MMSE检测的频谱效率"""
    Nt = 4
    Nr = 4
    snr_db_range = np.arange(0, 31, 5)
    num_trials = 1000
    
    se_zf = np.zeros(len(snr_db_range))
    se_mmse = np.zeros(len(snr_db_range))
    
    for snr_idx, snr_db in enumerate(snr_db_range):
        snr = 10 ** (snr_db / 10)
        sum_se_zf = 0
        sum_se_mmse = 0
        
        for _ in range(num_trials):
            # 生成信道矩阵
            H = generate_flat_rayleigh_channel(Nt, Nr)
            
            # 计算ZF检测的频谱效率
            H_inv_zf = np.linalg.pinv(H)
            Sigma_zf = H_inv_zf @ H_inv_zf.conj().T
            se_zf_trial = 2 * np.log2(np.linalg.det(np.eye(Nt) + snr * Sigma_zf)).real
            sum_se_zf += se_zf_trial
            
            # 计算MMSE检测的频谱效率
            I = np.eye(Nt)
            H_H = H.conj().T @ H
            H_inv_mmse = np.linalg.inv(H_H + I/snr) @ H.conj().T
            Sigma_mmse = H_inv_mmse @ H_inv_mmse.conj().T
            se_mmse_trial = 2 * np.log2(np.linalg.det(np.eye(Nt) + snr * Sigma_mmse)).real
            sum_se_mmse += se_mmse_trial
        
        # 计算平均频谱效率
        se_zf[snr_idx] = sum_se_zf / num_trials
        se_mmse[snr_idx] = sum_se_mmse / num_trials
    
    # 绘制频谱效率曲线
    plt.figure(figsize=(10, 6))
    plt.plot(snr_db_range, se_zf, 'o-', label='ZF Detection')
    plt.plot(snr_db_range, se_mmse, 's-', label='MMSE Detection')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Spectral Efficiency (bps/Hz)')
    plt.title('4x4 MIMO系统中ZF和MMSE检测的频谱效率')
    plt.grid(True)
    plt.legend()
    plt.savefig('mimo_spectral_efficiency.png')
    plt.close()
    
    print("MIMO检测频谱效率分析完成，结果已保存为mimo_spectral_efficiency.png")

# =====================================
# 第二部分：频率选择性瑞利衰落信道仿真
# =====================================

def generate_frequency_selective_channel(Nt, Nr, L, delay_spread):
    """
    生成频率选择性瑞利衰落信道
    
    参数:
    Nt: 发射天线数
    Nr: 接收天线数
    L: 多径数
    delay_spread: 时延扩展
    
    返回:
    H_freq: 频域信道矩阵
    """
    # 生成各径的时延和增益
    delays = np.sort(np.random.rand(L) * delay_spread)
    gains = (np.random.randn(Nr, Nt, L) + 1j * np.random.randn(Nr, Nt, L)) / np.sqrt(2)
    
    # 归一化增益，使总功率为1
    gains = gains / np.sqrt(np.sum(np.abs(gains)**2, axis=2, keepdims=True))
    
    return delays, gains

def analyze_frequency_selective_channel():
    """分析4x4 MIMO信道的频率选择性特性"""
    Nt = 4
    Nr = 4
    L = 8  # 多径数
    delay_spread = 10  # 时延扩展
    N = 128  # FFT大小
    
    # 生成频率选择性信道
    delays, gains = generate_frequency_selective_channel(Nt, Nr, L, delay_spread)
    
    # 计算频域信道响应
    H_freq = np.zeros((Nr, Nt, N), dtype=complex)
    f = np.fft.fftfreq(N)
    
    for k in range(N):
        for l in range(L):
            H_freq[:,:,k] += gains[:,:,l] * np.exp(-1j * 2 * np.pi * f[k] * delays[l])
    
    # 计算每个子载波的平均信道增益
    avg_gain = np.mean(np.abs(H_freq)**2, axis=(0,1))
    
    # 绘制频率响应
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(N), 10*np.log10(avg_gain))
    plt.xlabel('Subcarrier Index')
    plt.ylabel('Channel Gain (dB)')
    plt.title('4x4 MIMO信道的频率选择性特性')
    plt.grid(True)
    plt.savefig('frequency_selective_response.png')
    plt.close()
    
    print("频率选择性信道分析完成，结果已保存为frequency_selective_response.png")

def ofdm_modulate(complex_symbols, N, cp_length):
    """
    OFDM调制
    
    参数:
    complex_symbols: 输入复数符号
    N: FFT大小
    cp_length: 循环前缀长度
    
    返回:
    ofdm_signal: OFDM调制后的时域信号
    """
    # 确保符号数量等于FFT大小
    assert len(complex_symbols) == N, "符号数量必须等于FFT大小"
    
    # IFFT操作，从频域转换到时域
    time_signal = np.fft.ifft(complex_symbols) * np.sqrt(N)
    
    # 添加循环前缀
    cp = time_signal[-cp_length:]
    ofdm_signal = np.concatenate([cp, time_signal])
    
    return ofdm_signal

def ofdm_demodulate(ofdm_signal, N, cp_length):
    """
    OFDM解调
    
    参数:
    ofdm_signal: OFDM时域信号
    N: FFT大小
    cp_length: 循环前缀长度
    
    返回:
    complex_symbols: 解调后的频域符号
    """
    # 移除循环前缀
    time_signal = ofdm_signal[cp_length:]
    
    # FFT操作，从时域转换到频域
    complex_symbols = np.fft.fft(time_signal) / np.sqrt(N)
    
    return complex_symbols

def analyze_ofdm_channel():
    """分析OFDM系统频域信道特性"""
    N = 128  # FFT大小
    cp_length = 16  # 循环前缀长度
    L = 8  # 多径数
    delay_spread = 10  # 时延扩展
    
    # 生成随机QPSK符号
    bits = np.random.randint(0, 2, N*2)
    complex_symbols = qpsk_modulate(bits)
    
    # 生成频率选择性信道
    delays, gains = generate_frequency_selective_channel(1, 1, L, delay_spread)
    
    # OFDM调制
    ofdm_signal = ofdm_modulate(complex_symbols, N, cp_length)
    
    # 通过信道传输
    received_signal = np.zeros(len(ofdm_signal), dtype=complex)
    for l in range(L):
        # 计算时延对应的采样点
        delay_samples = int(round(delays[l] * len(ofdm_signal) / delay_spread))
        if delay_samples < len(ofdm_signal):
            received_signal[delay_samples:] += gains[0, 0, l] * ofdm_signal[:len(ofdm_signal)-delay_samples]
    
    # 添加噪声
    snr_db = 20
    snr = 10 ** (snr_db / 10)
    noise_power = 1 / snr
    noise = (np.random.randn(len(received_signal)) + 1j * np.random.randn(len(received_signal))) * np.sqrt(noise_power/2)
    received_signal += noise
    
    # OFDM解调
    received_symbols = ofdm_demodulate(received_signal, N, cp_length)
    
    # 估计信道频率响应
    h_est = received_symbols / complex_symbols
    
    # 计算理论信道频率响应
    h_theory = np.zeros(N, dtype=complex)
    f = np.fft.fftfreq(N)
    for k in range(N):
        for l in range(L):
            h_theory[k] += gains[0, 0, l] * np.exp(-1j * 2 * np.pi * f[k] * delays[l])
    
    # 绘制信道频率响应
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(N), 10*np.log10(np.abs(h_theory)**2), label='理论')
    plt.plot(np.arange(N), 10*np.log10(np.abs(h_est)**2), 'o', label='估计', markersize=3)
    plt.xlabel('子载波索引')
    plt.ylabel('信道增益 (dB)')
    plt.title('OFDM系统频域信道响应')
    plt.legend()
    plt.grid(True)
    
    # 绘制相位响应
    plt.subplot(2, 1, 2)
    plt.plot(np.arange(N), np.angle(h_theory), label='理论')
    plt.plot(np.arange(N), np.angle(h_est), 'o', label='估计', markersize=3)
    plt.xlabel('子载波索引')
    plt.ylabel('相位 (rad)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('ofdm_channel_response.png')
    plt.close()
    
    print("OFDM信道分析完成，结果已保存为ofdm_channel_response.png")

def analyze_mimo_ofdm_mmse():
    """分析MIMO-OFDM系统中MMSE检测的性能"""
    Nt = 4  # 发射天线数
    Nr = 4  # 接收天线数
    N = 128  # FFT大小
    cp_length = 16  # 循环前缀长度
    L = 8  # 多径数
    delay_spread = 10  # 时延扩展
    snr_db_range = np.arange(0, 21, 2)
    num_bits = 10000
    ber_mimo_ofdm = np.zeros(len(snr_db_range))
    
    for snr_idx, snr_db in enumerate(snr_db_range):
        snr = 10 ** (snr_db / 10)
        num_errors = 0
        
        # 生成随机比特
        bits = np.random.randint(0, 2, num_bits)
        
        # 分成多个OFDM符号处理
        bits_per_symbol = N * 2  # 每个OFDM符号的比特数 (QPSK)
        num_symbols = num_bits // bits_per_symbol
        
        for sym_idx in range(num_symbols):
            # 提取当前OFDM符号的比特
            sym_bits = bits[sym_idx*bits_per_symbol : (sym_idx+1)*bits_per_symbol]
            
            # QPSK调制
            complex_symbols = qpsk_modulate(sym_bits)
            
            # 对每个发射天线生成OFDM信号
            ofdm_signals = np.zeros((Nt, N + cp_length), dtype=complex)
            for tx_ant in range(Nt):
                ofdm_signals[tx_ant] = ofdm_modulate(complex_symbols, N, cp_length)
            
            # 生成频率选择性MIMO信道
            delays, gains = generate_frequency_selective_channel(Nt, Nr, L, delay_spread)
            
            # 通过信道传输
            received_signals = np.zeros((Nr, N + cp_length), dtype=complex)
            for rx_ant in range(Nr):
                for tx_ant in range(Nt):
                    # 对每个多径分量
                    for l in range(L):
                        # 计算时延对应的采样点
                        delay_samples = int(round(delays[l] * (N + cp_length) / delay_spread))
                        if delay_samples < (N + cp_length):
                            received_signals[rx_ant, delay_samples:] += gains[rx_ant, tx_ant, l] * ofdm_signals[tx_ant, : (N + cp_length) - delay_samples]
            
            # 添加噪声
            noise_power = 1 / snr
            noise = (np.random.randn(Nr, N + cp_length) + 1j * np.random.randn(Nr, N + cp_length)) * np.sqrt(noise_power/2)
            received_signals += noise
            
            # OFDM解调
            received_symbols = np.zeros((Nr, N), dtype=complex)
            for rx_ant in range(Nr):
                received_symbols[rx_ant] = ofdm_demodulate(received_signals[rx_ant], N, cp_length)
            
            # 估计每个子载波的信道矩阵
            H_est = np.zeros((Nr, Nt, N), dtype=complex)
            for k in range(N):
                for rx_ant in range(Nr):
                    for tx_ant in range(Nt):
                        for l in range(L):
                            H_est[rx_ant, tx_ant, k] += gains[rx_ant, tx_ant, l] * np.exp(-1j * 2 * np.pi * k * delays[l] / N)
            
            # MMSE检测每个子载波
            detected_symbols = np.zeros(N, dtype=complex)
            for k in range(N):
                H_k = H_est[:,:,k]
                y_k = received_symbols[:,k]
                
                # MMSE检测
                s_hat_k = mmse_detection(y_k, H_k, snr)
                
                # 取第一个发射天线的检测结果 (简化处理)
                detected_symbols[k] = s_hat_k[0]
            
            # QPSK解调
            detected_bits = qpsk_demodulate(detected_symbols)
            
            # 计算错误数
            num_errors += np.sum(detected_bits != sym_bits)
        
        # 计算BER
        ber_mimo_ofdm[snr_idx] = num_errors / num_bits
    
    # 绘制BER曲线
    plt.figure(figsize=(10, 6))
    plt.semilogy(snr_db_range, ber_mimo_ofdm, 'o-')
    plt.xlabel('SNR (dB)')
    plt.ylabel('BER')
    plt.title('MIMO-OFDM系统中MMSE检测的BER性能')
    plt.grid(True)
    plt.savefig('mimo_ofdm_ber.png')
    plt.close()
    
    print("MIMO-OFDM MMSE检测分析完成，结果已保存为mimo_ofdm_ber.png")

def simulate_mimo_ofdm():
    """MIMO-OFDM系统仿真"""
    # 仿真参数
    N_tx = 2  # 发射天线数
    N_rx = 2  # 接收天线数
    N = 64  # OFDM子载波数
    CP = 16  # 循环前缀长度
    N_bits = 10000  # 仿真比特数
    SNR_dB = np.arange(-10, 21, 2)  # SNR范围：-10到20dB
    
    # 存储结果
    ber_zf = []
    ber_mmse = []
    ber_ml = []
    
    # 生成随机比特
    bits = np.random.randint(0, 2, (N_tx, N_bits))
    
    # BPSK调制
    symbols = 2*bits - 1
    
    # 对每个SNR进行仿真
    for snr in SNR_dB:
        # 添加噪声
        noise_power = 10**(-snr/10)
        noise = np.sqrt(noise_power/2) * (
            np.random.randn(N_rx, N_bits) + 1j*np.random.randn(N_rx, N_bits)
        )
        
        # 生成MIMO信道
        H = (np.random.randn(N_rx, N_tx, N_bits) + 
             1j*np.random.randn(N_rx, N_tx, N_bits)) / np.sqrt(2)
        
        # 通过MIMO信道
        received = np.zeros((N_rx, N_bits), dtype=complex)
        for i in range(N_bits):
            received[:, i] = H[:, :, i] @ symbols[:, i] + noise[:, i]
        
        # 1. ZF检测
        zf_symbols = np.zeros_like(symbols)
        for i in range(N_bits):
            H_pinv = np.linalg.pinv(H[:, :, i])
            zf_symbols[:, i] = H_pinv @ received[:, i]
        zf_bits = (zf_symbols.real > 0).astype(int)
        ber_zf.append(np.mean(bits != zf_bits))
        
        # 2. MMSE检测
        mmse_symbols = np.zeros_like(symbols)
        for i in range(N_bits):
            H_H = H[:, :, i].conj().T
            H_H_H = H_H @ H[:, :, i]
            mmse_symbols[:, i] = np.linalg.inv(H_H_H + noise_power * np.eye(N_tx)) @ H_H @ received[:, i]
        mmse_bits = (mmse_symbols.real > 0).astype(int)
        ber_mmse.append(np.mean(bits != mmse_bits))
        
        # 3. ML检测（使用穷举搜索）
        ml_symbols = np.zeros_like(symbols)
        possible_symbols = np.array([-1, 1])
        for i in range(N_bits):
            min_dist = float('inf')
            for s1 in possible_symbols:
                for s2 in possible_symbols:
                    s = np.array([s1, s2])
                    dist = np.linalg.norm(received[:, i] - H[:, :, i] @ s)
                    if dist < min_dist:
                        min_dist = dist
                        ml_symbols[:, i] = s
        ml_bits = (ml_symbols.real > 0).astype(int)
        ber_ml.append(np.mean(bits != ml_bits))
        
        # 打印结果
        print(f"SNR = {snr:2d} dB, ZF BER = {ber_zf[-1]:.6f}, MMSE BER = {ber_mmse[-1]:.6f}, ML BER = {ber_ml[-1]:.6f}")
    
    # 绘制性能曲线
    plt.figure(figsize=(10, 6))
    plt.semilogy(SNR_dB, ber_zf, 'o-', label='ZF检测')
    plt.semilogy(SNR_dB, ber_mmse, 's-', label='MMSE检测')
    plt.semilogy(SNR_dB, ber_ml, '^-', label='ML检测')
    plt.xlabel('信噪比 (dB)')
    plt.ylabel('误码率 (BER)')
    plt.title('2x2 MIMO-OFDM系统在不同检测方式下的性能')
    plt.grid(True, which='both')
    plt.legend()
    plt.tight_layout()
    plt.savefig('stage4_mimo_ofdm_performance.png')
    plt.close()
    
    # 分析空间分集增益
    plt.figure(figsize=(10, 6))
    plt.plot(SNR_dB, np.array(ber_zf)/np.array(ber_ml), 'o-', label='ZF vs ML')
    plt.plot(SNR_dB, np.array(ber_mmse)/np.array(ber_ml), 's-', label='MMSE vs ML')
    plt.xlabel('信噪比 (dB)')
    plt.ylabel('性能差距倍数')
    plt.title('不同检测方式与ML检测的性能差距')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('stage4_detection_comparison.png')
    plt.close()
    
    print("MIMO-OFDM系统仿真完成")

if __name__ == "__main__":
    print("开始MIMO系统仿真评估...")
    
    # 平坦瑞利衰落部分
    print("\n=== 平坦瑞利衰落信道仿真 ===")
    print("1. 分析空间相关性...")
    analyze_spatial_correlation()
    
    print("2. 分析STBC编码的BER性能...")
    analyze_stbc_ber()
    
    print("3. 分析4x4 MIMO系统中ZF和MMSE检测的频谱效率...")
    analyze_mimo_detection()
    
    # 频率选择性瑞利衰落部分
    print("\n=== 频率选择性瑞利衰落信道仿真 ===")
    print("1. 分析4x4 MIMO信道的频率选择性特性...")
    analyze_frequency_selective_channel()
    
    print("2. 分析OFDM系统频域信道特性...")
    analyze_ofdm_channel()
    
    print("3. 分析MIMO-OFDM系统中MMSE检测的性能...")
    analyze_mimo_ofdm_mmse()
    
    print("\n仿真评估完成！所有结果已保存为PNG文件。")
    
    simulate_mimo_ofdm()    