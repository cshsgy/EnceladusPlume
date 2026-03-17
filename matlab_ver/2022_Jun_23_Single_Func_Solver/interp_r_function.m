function [r,phi] = interp_r_function(d,w)
    load('/home/shc/Enceladus/2021_25Jul_ParallelCrackDyn/r_rec_large_delta.mat','Tb','depth','delta','phi_rec','r_rec');
    % Assume that only interpolation exists. Any extrapolation would cause
    % error.

    % Optimized for T=273.15 K only
    iT = 1;
    id = sum(depth<d);
    iw = sum(delta<=w);
    if iw==0
        iw = 1; % extrap for smaller width...
    end
    if id*iT*iw==0 || id==length(depth) || iw==length(w)
        r = nan;
        phi = nan;
        return;
    end
    dd = (d-depth(id))/(depth(id+1)-depth(id));
    wd = (w-delta(iw))/(delta(iw+1)-delta(iw));
    
    % Find the r
    c00 = r_rec(iw,id,end);
    c01 = r_rec(iw+1,id,end);
    c10 = r_rec(iw,id+1,end);
    c11 = r_rec(iw+1,id+1,end);
    c0 = c00*(1-dd)+c10*dd;
    c1 = c01*(1-dd)+c11*dd;
    r = c0*(1-wd)+c1*wd;
    
    % Find the top evap
    c00 = phi_rec(iw,id,end);
    c01 = phi_rec(iw+1,id,end);
    c10 = phi_rec(iw,id+1,end);
    c11 = phi_rec(iw+1,id+1,end);
    c0 = c00*(1-dd)+c10*dd;
    c1 = c01*(1-dd)+c11*dd;
    phi = c0*(1-wd)+c1*wd;
    
end