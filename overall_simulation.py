import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rayleigh, rice, nakagami
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

def run_stage1_simulation():
    """阶段1：无线信道仿真"""
    print("\n=== 阶段1：无线信道仿真 ===")
    from main import WirelessChannel
    
    # 初始化信道模型
    channel = WirelessChannel()
    
    # 路径损耗计算
    d = np.logspace(2, 5, 1000)  # 100m到100km
    pl = channel.hata_path_loss(d)
    
    # 阴影衰落仿真
    shadow = channel.shadow_fading(size=1000)
    
    # 快衰落仿真
    fading_rayleigh = channel.jakes_fading(model='rayleigh')
    fading_rice = channel.jakes_fading(model='rice', K=10)
    fading_nakagami = channel.jakes_fading(model='nakagami', m=2)
    
    # 绘制结果
    plt.figure(figsize=(15, 10))
    
    plt.subplot(221)
    plt.semilogx(d, pl)
    plt.title("路径损耗（Hata模型）")
    plt.xlabel("距离（米）")
    plt.ylabel("路径损耗（dB）")
    plt.grid(True)
    
    plt.subplot(222)
    plt.plot(shadow)
    plt.title("阴影衰落")
    plt.xlabel("样本索引")
    plt.ylabel("衰落（dB）")
    plt.grid(True)
    
    plt.subplot(223)
    plt.plot(np.abs(fading_rayleigh), label='Rayleigh')
    plt.plot(np.abs(fading_rice), label='Rice (K=10)')
    plt.plot(np.abs(fading_nakagami), label='Nakagami (m=2)')
    plt.title("快衰落包络")
    plt.xlabel("样本索引")
    plt.ylabel("幅度")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('stage1_results.png')
    plt.close()
    
    print("阶段1仿真完成，结果已保存为stage1_results.png")

def run_stage2_simulation():
    """阶段2：调制与编码仿真"""
    print("\n=== 阶段2：调制与编码仿真 ===")
    import test2.main1 as stage2_main1
    import test2.main2 as stage2_main2
    import test2.main3 as stage2_main3
    
    # 运行RSC滤波器和BPSK调制仿真
    stage2_main1.simulate_rrc_bpsk()
    
    # 运行BPSK/QPSK/16QAM调制比较
    stage2_main2.simulate_modulation_comparison()
    
    # 运行卷积编码和交织仿真
    stage2_main3.simulate_coding_interleaving()
    
    print("阶段2仿真完成")

def run_stage3_simulation():
    """阶段3：OFDM系统仿真"""
    print("\n=== 阶段3：OFDM系统仿真 ===")
    import test3.main1 as stage3_main1
    import test3.main2 as stage3_main2
    import test3.main3 as stage3_main3
    import test3.main_p3 as stage3_main_p3
    
    # 运行OFDM基本功能仿真
    stage3_main1.simulate_ofdm_basic()
    
    # 运行OFDM信道均衡仿真
    stage3_main2.simulate_ofdm_equalization()
    
    # 运行OFDM性能分析
    stage3_main3.simulate_ofdm_performance()
    
    # 运行OFDM其他功能
    stage3_main_p3.simulate_ofdm_additional()
    
    print("阶段3仿真完成")

def run_stage4_simulation():
    """阶段4：MIMO-OFDM系统仿真"""
    print("\n=== 阶段4：MIMO-OFDM系统仿真 ===")
    import test4.main as stage4_main
    
    # 运行MIMO-OFDM系统仿真
    stage4_main.simulate_mimo_ofdm()
    
    print("阶段4仿真完成")

def run_complete_system_simulation():
    """运行完整的无线通信系统仿真"""
    print("开始完整无线通信系统仿真...")
    
    # 运行各个阶段的仿真
    run_stage1_simulation()
    run_stage2_simulation()
    run_stage3_simulation()
    run_stage4_simulation()
    
    # 生成系统性能总结报告
    generate_performance_report()
    
    print("\n完整无线通信系统仿真完成！")

def generate_performance_report():
    """生成系统性能总结报告"""
    print("\n=== 系统性能总结报告 ===")
    print("1. 无线信道性能")
    print("   - 路径损耗：使用Hata模型，适用于城市、郊区和农村环境")
    print("   - 阴影衰落：使用对数正态分布模型")
    print("   - 快衰落：支持Rayleigh、Rice和Nakagami模型")
    
    print("\n2. 调制与编码性能")
    print("   - 支持BPSK、QPSK和16QAM调制")
    print("   - 使用RSC滤波器和卷积编码")
    print("   - 实现了块交织和Viterbi译码")
    
    print("\n3. OFDM系统性能")
    print("   - 支持多载波传输")
    print("   - 实现了循环前缀和信道均衡")
    print("   - 包含完整的OFDM收发链路")
    
    print("\n4. MIMO-OFDM系统性能")
    print("   - 支持多天线传输")
    print("   - 实现了空间分集和复用")
    print("   - 包含完整的MIMO-OFDM收发链路")
    
    print("\n所有仿真结果已保存为相应的图片文件")

if __name__ == "__main__":
    run_complete_system_simulation()