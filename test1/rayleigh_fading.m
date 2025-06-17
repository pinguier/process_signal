function h = rayleigh_fading(t, fd)
    N = 8; % 典型Jakes模型参数
    h = zeros(size(t));
    for n = 1:N
        theta_n = pi*n/N;
        h = h + cos(2*pi*fd*cos(theta_n)*t + rand()*2*pi);
    end
    h = sqrt(2/N) * h;
end
