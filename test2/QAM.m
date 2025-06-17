% 首先清除所有现有的数据。
clc;  % 清除命令窗口
clear all;  % 清除工作空间
close all;  % 关闭所有其他可操作的窗口
 
% 生成16 QAM星座图作为复数
b = 4; % 每个符号中的比特数（它是偶数）
M = 2^b; % 星座中的点数 = 2^4 = 16
% 因为我们有一个4×4的星座图，它是对称的，我们可以简单地选择
% 点 -3, -1, 1, 3 。
x1 = -(b-1):2:(b-1); % 由于我们有b个符号，b/2位于0的左侧，b/2位于0的右侧。
constellation = x1 + 1i*x1.';  % 将实部和虚部相加
k = double(1.0)/double(sqrt(10)); % 正规化因子
constellation = k * constellation; % 正规化星座图，使其功率为单位。
 
% 为非格雷编码在星座图中的映射（二进制到十进制按位置映射）
% 在二进制                          在十进制
% 0011 0111 1011 1111      |    3  7  11  15
% 0010 0110 1010 1110      |    2  6  10  14
% 0001 0101 1001 1101      |    1  5  9   13
% 0000 0100 1000 1100      |    0  4  8   12
 
% 为格雷编码在星座图中的映射
% 在二进制                          在十进制
% 0010 0110 1110 1010      |    2  6  14  10
% 0011 0111 1111 1011      |    3  7  15  11
% 0001 0101 1101 1001      |    1  5  13  9
% 0000 0100 1100 1000      |    0  4  12  8
 
% 因此我们使用数组来映射它们
gre = [0 1 3 2 4 5 7 6 12 13 15 14 8 9 11 10]; % 用于映射非格雷编码和格雷编码星座点
 
ninputs = 10000; % 表示用于仿真使用的符号数量。
input = zeros(1, ninputs);
for k = 1:ninputs % 循环生成4比特随机输入
    input(k) = randi([0, (2^4-1)]); % 随机生成一个4比特数，范围从0到15（包括0和15）。
end
binc = constellation(input(:) + 1); % 将具有非格雷编码的星座符号
input_gray = gre(input(:) + 1); % 获取与相同星座输入相对应的格雷编码输入。
 
snr = 0:1:10; % 改变信噪比从0到10dB。
% 我们假设输入信号是所有输入c中的点。
decisions_bin = zeros(1, ninputs);
number_snrs = length(snr); % 需要检查的信噪比值的数量
Ber1 = zeros(number_snrs, 1); % 估计每个信噪比下的比特误码率，并将其累加到估计中
Ber2 = zeros(number_snrs, 1); % 估计每个信噪比下的比特误码率，并将其累加到估计中
 
% 仿真的开始。
for k = 1:number_snrs % 信噪比循环
    snr_now = snr(k); % 当前测试的信噪比。
    ebno = 10^(snr_now / 10); % 将信噪比从dB转换为十进制单位。
    sigma = sqrt(1 / ebno); % 对应的噪声方差。
    % 向我们的符号添加2D高斯噪声。
    receivedbin = binc + (sigma * randn(ninputs, 1) + 1i * sigma * randn(ninputs, 1)) / sqrt(10); % 添加复数白高斯噪声到输入信号，并适当缩放。
    decisions = zeros(ninputs, 1); % 初始化决策变量，与所有n个符号对应。
    for n = 1:ninputs
        distancesbin = abs(receivedbin(n) - constellation); % 计算每个信号点与星座图中每个点的距离。
        [min_dist_bin, decisions_bin(n)] = min(distancesbin(:)); % 最小距离的星座点即为信号。
    end
    
    decisions_gray = gre(decisions_bin); % 将解码信号映射回格雷编码输入，以比较格雷编码的错误。
    % decisions_bin是索引值，它们对应于某个
    % decisions_gray值。
    decisions_bin = decisions_bin - 1; % 使其范围从0到15。
    
    % 计算比特误码率
    num = zeros(1, ninputs); % 为了加快代码执行速度
    for s = 1:length(input)
        d_bin = de2bi(decisions_bin(s), 4); % 获取零填充的4比特二进制字符串，便于比较。
        i_bin = de2bi(input(s), 4);  % 获取零填充的4比特二进制字符串，便于比较。
        biterror = 0;   % 计数每个比特的错误
        for t = 1:4
            if d_bin(t) ~= i_bin(t)
                biterror = biterror + 1;  % 添加每个错误决定的比特。
            end
            num(s) = biterror; % 存储每个单词的总比特错误
        end
    end
    error = num; % 获取非格雷编码信号的比特错误数组。
    
    for s = 1:length(input_gray)
        d_bin = de2bi(decisions_gray(s), 4); % 获取零填充的4比特二进制字符串，便于比较。
        i_bin = de2bi(input_gray(s), 4);  % 获取零填充的4比特二进制字符串，便于比较。
        biterror = 0;   % 计数每个比特的错误
        for t = 1:4
            if d_bin(t) ~= i_bin(t)
                biterror = biterror + 1;  % 添加每个错误决定的比特。
            end
            num(s) = biterror; % 存储每个单词的总比特错误
        end
    end
    error_gray = num; % 获取格雷编码信号的比特错误数组。
    
    Ber1(k) = Ber1(k) + sum(error) / ninputs; % 给出比特误码率。
    Ber2(k) = Ber2(k) + sum(error_gray) / ninputs; % 给出比特误码率。
end
 
% 绘制比特误码率（BER）图。
figure;
semilogy(snr, Ber1); % 绘制信噪比与比特误码率的关系。
hold on; % 在同一图中添加更多数据
semilogy(snr, Ber2); % 绘制信噪比与比特误码率的关系。
hold on;
% 由于对于16 QAM平均邻居数量为3，我们乘以Q函数的3倍
semilogy(snr, 3 * qfunc(sqrt((10.^(snr/10))))); % 使用Q函数绘制理论比特误码率。
legend("实验比特误码率", "实验比特误码率（使用格雷编码）", "使用Q函数的理论值"); % 添加图例
xlabel("信噪比 (dB)"); % 添加信噪比标签到x轴
ylabel("比特误码率 (BER)"); % 添加比特误码率标签到y轴。这是每符号的BER。
title("16 QAM的比特误码率图");
 
 
 
