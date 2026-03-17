function [result] = pvpt(zs, w, dwdt, dwdt2)
% Essentially partial v partial t peeling off dv0/dt
npts = length(zs);
integrand = zeros(npts,1);
result = 0;
for i=1:npts
    integrand(i) = dwdt^2/w^2-dwdt2/w;
    if i>1
        result = result + 0.5*(integrand(i-1)+integrand(i))*(zs(i)-zs(i-1));
    end
end
end
