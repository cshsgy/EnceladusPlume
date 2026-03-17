function [w] = width_new(z, t, wmaxmin, wmin)
% Take z, t, give the width as function of it
% In the unit of meters
% **It should tolerate some t smaller than zero and z smaller than -L and
% **larger than 0. Must ensure it NEVER goes below zero

% Straight Crack
orbitper = 1.37*86400;
g = 0.11;
L = 20000;
eps = 0.5*(wmaxmin-1);
omega = 2*pi/orbitper; % *sqrt(L/g)
w = 1 + eps*(1-cos(omega*t));
w = w * wmin;

% w = 1.0 * exp(-t); % The "simple testing case"

if w<1e-8
    w=1e-8;
end
end

