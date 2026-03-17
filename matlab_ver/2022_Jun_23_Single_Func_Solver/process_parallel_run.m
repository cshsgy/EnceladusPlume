%% Plot the lines...
foldern = './run_results/';
fils = dir(foldern);
figure;
hold on;
legends = {};
for i=1:length(fils)
	if ~contains(fils(i).name,'.mat')
		continue;
	end
	filn = strcat(foldern,fils(i).name);
	load(filn,'Period','time_rec','phi_rec');
	plot(time_rec/Period,smooth(MaxFilter(phi_rec,3)));
	tmp = fils(i).name;
	tmp = tmp(1:end-4);
	legends{end+1} = strrep(tmp,'_','-');
end
xlim([19 20]);
xlabel('Time (Day)');
ylabel('Mass Flux (kg m^{-1}s^{-1})');
legend(legends,'Location','bestoutside');
hold off;
saveas(gcf,'MassFlux.png');

figure;
hold on;
legends = {};
for i=1:length(fils)
	if ~contains(fils(i).name,'.mat')
		continue;
	end
	filn = strcat(foldern,fils(i).name);
	load(filn,'Period','time_rec','r_rec');
	plot(time_rec/Period,smooth(MaxFilter(r_rec,5)));
	tmp = fils(i).name;
	tmp = tmp(1:end-4);
	legends{end+1} = strrep(tmp,'_','-');
end
xlim([19 20]);
xlabel('Time (Day)');
ylabel('P_B/P_v');
legend(legends,'Location','bestoutside');
hold off;
saveas(gcf,'Ratio.png');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Plot the contours...
r = [];
w = [];
mp = [];
sp = [];
for i=1:length(fils)
	if ~contains(fils(i).name,'.mat')
		continue;
	end
	tmp = strrep(fils(i).name,'_wmin_',' ');
	tmp = strrep(tmp,'.mat','');
	tmp = strrep(tmp,'r_','');
	str_cell = split(tmp);
	r(end+1) = str2num(str_cell{1});
	w(end+1) = str2num(str_cell{2});
	filn = strcat(foldern,fils(i).name);
	load(filn,'Period','time_rec','phi_rec');
	phi_rec = MaxFilter(phi_rec,5);
	time_rec = time_rec/Period;
	phi_rec_1 = phi_rec((time_rec>19.3) & (time_rec<19.7));
	phi_rec_2 = phi_rec((time_rec>19.7) & (time_rec<20.0));
	phi_rec_3 = phi_rec((time_rec>19.0) & (time_rec<19.3));
	mp(end+1) = max(phi_rec_1);
	sp(end+1) = max(max(phi_rec_2),max(phi_rec_3));
end

% Reconstruct the 2d matrix
rid = sort(unique(r));
wid = sort(unique(w));
mp_mat = zeros(length(rid),length(wid));
sp_mat = zeros(length(rid),length(wid));
for i=1:length(r)
	rpos = find(rid==r(i));
	wpos = find(wid==w(i));
	mp_mat(rpos,wpos) = mp(i);
	sp_mat(rpos,wpos) = sp(i);
end
subplot(1,2,1);
contourf(wid,rid,mp_mat);
ylabel('\delta_{max}/\delta_{min}');
xlabel('Minimum width (m)');
title('Primary peak strength');
set(gca,'XScale','log');
colorbar;
subplot(1,2,2);
contourf(wid,rid,sp_mat);
ylabel('\delta_{max}/\delta_{min}');
xlabel('Minimum width (m)');
title('Secondary peak strength');
set(gca,'XScale','log');
colorbar;
saveas(gcf,'contours.png');

close all;