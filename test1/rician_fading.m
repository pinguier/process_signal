function h = rician_fading(t, fd, K)
    rayleigh = rayleigh_fading(t, fd);
    direct = sqrt(K/(K+1));
    scatter = sqrt(1/(K+1)) * rayleigh;
    h = direct + scatter;
end
