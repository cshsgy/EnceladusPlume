zs = 1:1:1000;
tic
[PhiTop,Tw,Ev,r] = GasDynamicsMarchInTime(0.1,zs,ones(1,length(zs))*270,3,2e6,0.113,0.002,0.1);
toc
% [~,~,M,Phi] = MikiModelFull(0.9,0.1,zs,ones(1,length(zs))*270,3,2e6,0.113,0.002,0.1)