function [T] = HeatDiffusionMarchInTime(z,x,h,d,T,Tw,Kappa,K,Te,Dx,Dt)
% Solve Heat Diffusion Equations using the calculated gas dynamics and
% evaporation strength data.
% Note: Tw should be found in the gas solver already.
% For parallelization, this solves for one lvl only.
nx = length(x);
dTdt = zeros(1,nx);
dTdt(1) = 0;
dTdt(end) = 0;
if z>h
    expose = 1;
else
    expose = 0;
end
if(expose==0)
    T(1) = 273.15;
else
    T(1) = Tw;
end
% Just the normal diffusion equation for everything within
for jj=2:(length(x)-1)
    dTdt(jj) = Kappa*(T(jj-1)-2*T(jj)+T(jj+1))/(Dx^2);
end
T = dTdt*Dt+T;
% The constant heat flux BC
Ts = SurfaceTemperature(Te,T(end),d,K);
T(end) = T(end-1)-(T(end)-Ts)/d*Dx;
end

