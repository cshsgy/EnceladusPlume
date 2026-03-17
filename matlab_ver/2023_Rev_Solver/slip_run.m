function [t_rec, w_rec, h_rec, phi_rec, dphi_rec] = slip_run(t_in, w_in, L)
    [w_rec,h_rec,t_rec] = main_func(w_in,t_in,L);
    disp(max(t_rec));
    D0 = L/10;
    P = 118800;
    % Keep one cycle
    while t_rec(end)>P/2
        t_rec = t_rec - P;
    end
    t_rec = t_rec + P;
    w_rec = w_rec(t_rec>0);
    h_rec = h_rec(t_rec>0);
    t_rec = t_rec(t_rec>0);
    phi_rec = zeros(length(t_rec),1);
    dphi_rec = zeros(length(t_rec),1);
    for i=1:length(t_rec)
        [r,phi,rho,phi0] = interp_r_function(273.15,D0-h_rec(i),w_rec(i));
        phi_rec(i) = phi * w_rec(i);
        dphi_rec(i) = (phi0 - phi) * w_rec(i);
    end
end