function h = nakagami_fading(t, m)
    omega = 1;
    h_mag = sqrt(gamrnd(m, omega/m, size(t)));
    h = h_mag .* exp(1i*2*pi*rand(size(t)));
end
