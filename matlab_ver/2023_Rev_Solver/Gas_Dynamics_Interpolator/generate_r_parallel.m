function generate_function(delta,depth)
Tb = [272,273.1501]; % Shorthand solution for interpolation to work
r_rec = zeros(length(delta),length(depth),length(Tb));
phi_rec = zeros(length(delta),length(depth),length(Tb));
rho_rec = zeros(length(delta),length(depth),length(Tb));
phi0_rec = zeros(length(delta),length(depth),length(Tb));

parfor i = 1:length(delta)
    tmp1 = zeros(length(depth),length(Tb));
    tmp2 = zeros(length(depth),length(Tb));
    tmp3 = zeros(length(depth),length(Tb));
    tmp4 = zeros(length(depth),length(Tb));
    for j=1:length(depth)
        for k = 1:length(Tb)
            [tmp1(j,k) tmp2(j,k) tmp3(j,k) tmp4(j,k)] = solve_r_function(Tb(k),depth(j),delta(i));
        end
    end
    r_rec(i,:,:) = tmp1;
    phi_rec(i,:,:) = tmp2;
    rho_rec(i,:,:) = tmp3;
    phi0_rec(i,:,:) = tmp4;
end
% Save the results for easier parallel access
save('r_rec.mat','r_rec','phi_rec','rho_rec','phi0_rec','Tb','delta','depth');
% NOTE: the phi_rec solved here is kg/s per meter square, so need to times delta in final calculation
end
