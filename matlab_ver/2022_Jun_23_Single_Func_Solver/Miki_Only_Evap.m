function evap = Miki_Only_Evap(width,height,L)
% Obtains miki-only 
depth = L-height;
evap = [];
for i=1:length(width)
    [~,phi] = interp_r_function(depth(i),width(i));
    evap(end+1) = phi;
end

end