function [result] = additional_term(zs,w,dwdt,dwdt2)
% This is the term on the RHS additionally peeled off from dvdt integral
npts = length(zs);
result = 0;
dadz = zeros(npts,1);
for i=1:npts
    dadz(i) = pvpt(zs(1:i),w,dwdt,dwdt2);
    if i>2
        result = result + 0.5*(dadz(i-1)+dadz(i))*(zs(i)-zs(i-1));
    end
end
end

