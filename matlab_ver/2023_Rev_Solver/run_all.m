tic;

t_in = load('Times.txt'); % 108
slips_in = load('Slip_Time_Functions.txt'); % 108*89

addpath('Gas_Dynamics_Interpolator');
addpath('Liquid_Dynamics_Solver');

t_in = t_in/360*118800;

w_base = 0.2; % Minimum width
L = 20000; % Equilibrium water depth

% Setup the grid of gas dynamics interpolation
n = 20;
w_max = max(slips_in(:))+w_base;
widths = exp(linspace(log(w_base)-0.01,log(w_max)+0.01,n));
depths = exp(linspace(log(10)-0.01,log(18000)+0.01,n));

% generate_r_parallel(widths,depths);

width_data = cell(size(slips_in,2),1);
level_data = cell(size(slips_in,2),1);
phi_data = cell(size(slips_in,2),1); % kg/(m*s)
dphi_data = cell(size(slips_in,2),1);
time_data = cell(size(slips_in,2),1);

parfor i=1:size(slips_in,2)
    [t_rec, w_rec, h_rec, phi_rec, dphi_rec] = slip_run(t_in(2:end), slips_in(2:end,i)+w_base, L);
    width_data{i} = w_rec;
    level_data{i} = h_rec;
    phi_data{i} = phi_rec;
    dphi_data{i} = dphi_rec;
    time_data{i} = t_rec;
end

save('test_run.mat','width_data','level_data','phi_data','dphi_data','time_data');

toc;