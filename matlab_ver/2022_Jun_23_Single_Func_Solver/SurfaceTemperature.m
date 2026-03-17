function [Ts] = SurfaceTemperature(Te,Tw,depth,K)
sigma = 5.67E-8; %Steffen-Boltzmann
syms x;
c = 2*K/(sigma*pi*depth);
tmp = double(vpasolve(x^4+c*x==Te^4+c*Tw,x));
Ts = -1;
for i=1:length(tmp)
    if isreal(tmp(i)) && tmp(i)>0
        Ts = tmp(i);
    end
end
end

