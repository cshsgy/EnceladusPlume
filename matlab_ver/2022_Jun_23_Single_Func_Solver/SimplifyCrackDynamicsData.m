function [t_rec,h_rec,w_rec] = SimplifyCrackDynamicsData(t_rec,Period,h_rec,w_rec)
% Extract the final Period seconds interval, from a start of full cycle
for i=length(t_rec)-1:-1:1
    if(mod(t_rec(i),Period)>mod(t_rec(i+1),Period))&&(t_rec(end)-t_rec(i)>=Period)
        break;
    end
end
for j=i+2:length(t_rec)
    if (mod(t_rec(j-1),Period)>mod(t_rec(j),Period))
        break;
    end
end
t_rec = mod(t_rec(i:j),Period);
t_rec(1) = t_rec(1)-Period;
t_rec(end) = t_rec(end)+Period;
h_rec = h_rec(i:j);
w_rec = w_rec(i:j);
end

