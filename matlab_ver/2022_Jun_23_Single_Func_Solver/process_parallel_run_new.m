%% Plot the lines...
foldern = './run_results/old_May15/';
filkey = 'wmin_0.010';
fils = dir(foldern);
figure;
hold on;
legends = {};
for i=1:length(fils)
	if ~contains(fils(i).name,'.mat') || ~contains(fils(i).name,filkey)
		continue;
	end
	filn = strcat(foldern,fils(i).name);
	load(filn,'Period','time_rec','phi_rec','width_rec');
	plot(time_rec/Period,smooth(phi_rec.*width_rec*520000,50),'linewidth',2); % 
	Ps = round(max(time_rec/Period));
	tmp = fils(i).name;
	tmp = tmp(1:end-4);
	tmp = tmp(1:6);
	tmp = strrep(tmp,'r','Width Ratio');
	legends{end+1} = strrep(tmp,'_','=');
end
xlim([Ps-3 Ps-2]);
xlabel('Time (Day)');
ylabel('Mass Flux (kg s^{-1})');
legend(legends,'Location','bestoutside');
title(strrep(filkey,'_','='));
hold off;
saveas(gcf,'MassFlux.png');

figure;
hold on;
legends = {};
for i=1:length(fils)
	if ~contains(fils(i).name,'.mat') || ~contains(fils(i).name,filkey)
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
xlim([Ps-2 Ps-1]);
xlabel('Time (Day)');
ylabel('P_B/P_v');
legend(legends,'Location','bestoutside');
title(strrep(filkey,'_','='));
hold off;
saveas(gcf,'Ratio.png');

figure;
hold on;
legends = {};
for i=1:length(fils)
	if ~contains(fils(i).name,'.mat') || ~contains(fils(i).name,filkey)
		continue;
	end
	filn = strcat(foldern,fils(i).name);
	load(filn,'Period','t_rec','h_rec');
	r_rec = r_rec(2:end);
	plot(t_rec/Period,h_rec,'linewidth',2); % 
	tmp = fils(i).name;
	tmp = tmp(1:end-4);
	tmp = tmp(1:6);
	tmp = strrep(tmp,'r','Width Ratio');
	legends{end+1} = strrep(tmp,'_','=');

end
xlabel('Time (Day)');
ylabel('Water Level (m)');
legend(legends,'Location','bestoutside');
title(strrep(filkey,'_','='));
hold off;
saveas(gcf,'WaterLevel.png');