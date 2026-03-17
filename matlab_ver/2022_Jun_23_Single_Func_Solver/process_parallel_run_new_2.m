%% Plot the lines...
foldern = './run_results/old_May15/';
fils = dir(foldern);
figure;
hold on;
legends = {};
for i=1:length(fils)
	if ~contains(fils(i).name,'.mat') || ~contains(fils(i).name,'wmin_0.004')
		continue;
	end
	filn = strcat(foldern,fils(i).name);
	load(filn,'Period','time_rec','phi_rec','width_rec');
	[phi_day,time_day] = accum_last_days(Period,time_rec,smooth(phi_rec.*width_rec*520000,100),5);
	plot(time_day,smooth(phi_day,100),'linewidth',2); % 
	tmp = fils(i).name;
	tmp = tmp(1:end-4);
	tmp = tmp(1:6);
	tmp = strrep(tmp,'r','Width Ratio');
	legends{end+1} = strrep(tmp,'_','=');
	ii = i;
end
xlabel('Time (Day)');
ylabel('Mass Flux (kg s^{-1})');
legend(legends,'Location','bestoutside');
title(strcat('W_{min}=',fils(ii).name(end-8:end-4),' m'))
hold off;
saveas(gcf,'MassFlux.png');

figure;
hold on;
legends = {};
for i=1:length(fils)
	if ~contains(fils(i).name,'.mat') || ~contains(fils(i).name,'wmin_0.002')
		continue;
	end
	filn = strcat(foldern,fils(i).name);
	load(filn,'Period','time_rec','r_rec');
	r_rec = r_rec(2:end);
	plot(time_rec/Period,smooth(r_rec,50),'linewidth',2); % 
	tmp = fils(i).name;
	tmp = tmp(1:end-4);
	tmp = tmp(1:6);
	tmp = strrep(tmp,'r','Width Ratio');
	legends{end+1} = strrep(tmp,'_','=');

end
Ps = round(max(time_rec/Period));
xlim([Ps-2 Ps-1]);
xlabel('Time (Day)');
ylabel('P_B/P_v');
legend(legends,'Location','bestoutside');
hold off;
saveas(gcf,'Ratio.png');
