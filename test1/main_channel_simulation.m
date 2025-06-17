clc; clear; close all;

%% 基础参数
fc = 2e9;                % 载波频率
c = 3e8;                 % 光速
lambda = c / fc;         % 波长
d = linspace(100, 5000, 100);  % 距离向量

%% 1. 路径损耗 - Hata模型
PL_hata = hata_model(fc/1e6, 30, 3, d);  % fc单位MHz

%% 2. 阴影衰落（对数正态）
shadowing = 10 * randn(1, length(d));   % σ=10dB

%% 3. 快衰落模型
fs = 1000;                % 采样率
t = (0:1/fs:10)';         % 模拟10秒时间序列

rayleigh = rayleigh_fading(t, 10);      % 多普勒10Hz
rician   = rician_fading(t, 10, 5);     % K=5
nakagami = nakagami_fading(t, 2);       % m=2

%% 4. 信道模型统一评估
models  = {'Rayleigh', 'Rician', 'Nakagami'};
signals = {rayleigh,  rician,    nakagami};
snr = 20;  % SNR for AWGN

for i = 1:length(models)
    model_name = models{i};
    signal     = signals{i};

    % 4.1 评估信道特性（时域相关、频谱、Tc/Bc）
    evaluate_fading_channel(signal, fs, model_name);

    % 4.2 加AWGN
    signal_noisy = awgn(signal, snr, 'measured');

    % 4.3 绘图展示加噪信号
    figure;
    plot(abs(signal_noisy));
    title([model_name ' 衰落 + AWGN']);
    xlabel('时间 (s)');
    ylabel('|信号|');
end

%% 5. 路径损耗 + 阴影衰落绘图
figure;
plot(d, PL_hata + shadowing);
title('路径损耗 (Hata) + 阴影衰落');
xlabel('距离 (m)');
ylabel('路径损耗 (dB)');
