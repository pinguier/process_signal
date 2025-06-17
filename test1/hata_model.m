function PL = hata_model(fc, ht, hr, d)
% fc单位MHz，ht/hr单位m，d单位m
    ahr = (1.1*log10(fc) - 0.7)*hr - (1.56*log10(fc) - 0.8);
    PL = 69.55 + 26.16*log10(fc) - 13.82*log10(ht) - ahr + ...
         (44.9 - 6.55*log10(ht))*log10(d/1000); % d单位km
end