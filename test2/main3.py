import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

def conv_encode(bits, g1=0b111, g2=0b101):
    """卷积编码 (约束长度3, (7,5)八进制)
    
    Args:
        bits: 输入比特序列
        g1: 第一个生成多项式（八进制）
        g2: 第二个生成多项式（八进制）
    
    Returns:
        编码后的比特序列
    """
    n = len(bits)
    state = 0
    encoded = []
    for i in range(n):
        state = ((state << 1) | bits[i]) & 0b11  # 2位寄存器
        out1 = bin((state << 1 | bits[i]) & g1).count('1') % 2
        out2 = bin((state << 1 | bits[i]) & g2).count('1') % 2
        encoded.extend([out1, out2])
    return np.array(encoded)

def block_interleave(bits, block_size):
    """块交织
    
    Args:
        bits: 输入比特序列
        block_size: 交织块大小
    
    Returns:
        交织后的比特序列
    """
    n = len(bits)
    num_blocks = int(np.ceil(n / block_size))
    padded = np.pad(bits, (0, num_blocks * block_size - n), 'constant')
    interleaved = padded.reshape((num_blocks, block_size)).T.flatten()
    return interleaved[:n]

def block_deinterleave(bits, block_size):
    """块去交织
    
    Args:
        bits: 输入比特序列
        block_size: 交织块大小
    
    Returns:
        去交织后的比特序列
    """
    n = len(bits)
    num_blocks = int(np.ceil(n / block_size))
    padded = np.pad(bits, (0, num_blocks * block_size - n), 'constant')
    deinterleaved = padded.reshape((block_size, num_blocks)).T.flatten()
    return deinterleaved[:n]

def viterbi_decode(received, g1=0b111, g2=0b101):
    """Viterbi译码 (硬判决，约束长度3，(7,5)八进制)
    
    Args:
        received: 接收到的比特序列
        g1: 第一个生成多项式（八进制）
        g2: 第二个生成多项式（八进制）
    
    Returns:
        译码后的比特序列
    """
    n = len(received) // 2
    K = 3
    n_states = 2**(K-1)
    path_metrics = np.full((n+1, n_states), np.inf)
    path_metrics[0, 0] = 0
    paths = np.zeros((n+1, n_states), dtype=int)
    
    for i in range(n):
        for state in range(n_states):
            for bit in [0, 1]:
                prev_state = ((state >> 1) | (bit << (K-2))) & (n_states-1)
                out1 = bin(((prev_state << 1) | bit) & g1).count('1') % 2
                out2 = bin(((prev_state << 1) | bit) & g2).count('1') % 2
                metric = (received[2*i] != out1) + (received[2*i+1] != out2)
                if path_metrics[i, prev_state] + metric < path_metrics[i+1, state]:
                    path_metrics[i+1, state] = path_metrics[i, prev_state] + metric
                    paths[i+1, state] = prev_state
    
    # 回溯
    state = np.argmin(path_metrics[-1])
    decoded = []
    for i in range(n, 0, -1):
        prev_state = paths[i, state]
        decoded.append((state >> (K-2)) & 1)
        state = prev_state
    return np.array(decoded[::-1])

def plot_eye_diagram(symbols, rx_eq, ebn0_dB):
    """绘制眼图
    
    Args:
        symbols: 发送符号
        rx_eq: 均衡后的接收信号
        ebn0_dB: 信噪比(dB)
    """
    samples_per_symbol = 1
    span = 2
    num_traces = 200
    
    plt.figure(figsize=(10, 6))
    for i in range(num_traces):
        start = i * samples_per_symbol
        end = start + span * samples_per_symbol
        if end <= len(rx_eq):
            plt.plot(np.arange(span * samples_per_symbol), rx_eq[start:end], color='b', alpha=0.3)
    plt.title(f'眼图 (Eb/N0={ebn0_dB}dB)')
    plt.xlabel('采样点')
    plt.ylabel('幅度')
    plt.grid(True)
    plt.savefig('stage2_eye_diagram.png')
    plt.close()

def simulate_coding_interleaving():
    """卷积编码和交织仿真"""
    # 仿真参数
    N = 10000  # 信息比特数
    EbN0_dB_range = np.arange(0, 11, 2)  # Eb/N0范围(dB)
    block_size = 20  # 交织块大小
    
    # 存储BER结果
    BER = []
    
    # 生成随机比特
    bits = np.random.randint(0, 2, N)
    
    # 卷积编码
    coded_bits = conv_encode(bits)
    
    # 块交织
    interleaved = block_interleave(coded_bits, block_size)
    
    # BPSK调制
    symbols = 1 - 2 * interleaved
    
    # 在不同Eb/N0下进行仿真
    for EbN0_dB in EbN0_dB_range:
        # Rayleigh平坦衰落信道
        h = np.sqrt(0.5) * (np.random.randn(len(symbols)) + 1j * np.random.randn(len(symbols)))
        EbN0 = 10**(EbN0_dB/10)
        noise_std = 1/np.sqrt(2*EbN0)
        noise = noise_std * np.random.randn(len(symbols))
        rx = h * symbols + noise
        
        # BPSK解调（已知信道）
        rx_eq = np.real(rx / h)
        demod_bits = (rx_eq < 0).astype(int)
        
        # 去交织
        deinterleaved = block_deinterleave(demod_bits, block_size)
        
        # Viterbi译码
        decoded = viterbi_decode(deinterleaved)
        
        # 计算误码率
        ber = np.sum(decoded[:N] != bits) / N
        BER.append(ber)
        print(f'Eb/N0={EbN0_dB}dB, BER={ber:.5e}')
        
        # 在Eb/N0=6dB时绘制眼图
        if EbN0_dB == 6:
            plot_eye_diagram(symbols, rx_eq, EbN0_dB)
    
    # 绘制BER曲线
    plt.figure(figsize=(10, 6))
    plt.semilogy(EbN0_dB_range, BER, 'o-')
    plt.xlabel('Eb/N0 (dB)')
    plt.ylabel('BER')
    plt.title('BER vs Eb/N0 (卷积码+交织+Rayleigh信道)')
    plt.grid(True)
    plt.savefig('stage2_coding_interleaving.png')
    plt.close()
    
    print("卷积编码和交织仿真完成")

if __name__ == "__main__":
    simulate_coding_interleaving()