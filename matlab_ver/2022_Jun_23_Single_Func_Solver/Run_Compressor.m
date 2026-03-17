% Compresses the calculation into smaller files...

foldern = './run_results/old_mar30/';
filto = 'Compressed_run_Mar30';
fils = dir(foldern);

WidthRatios = [];
MinWidths = [];
Phis = {};
BackPressures = {};
WidthRecs = {};
HeightRecs = {};
Evaporations = {};

for i=1:length(fils)
	if ~contains(fils(i).name,'.mat')
		continue;
	end
    tmp = fils(i).name;
    disp(tmp);
    tmp = strrep(tmp,'.mat','');
    tmp = split(tmp,'_');
	filn = strcat(foldern,fils(i).name);
	load(filn,'Period','time_rec','phi_rec','width_rec','t_rec','w_rec','r_rec','h_rec');
	[phi_day,time_day] = accum_last_days(Period,time_rec,smooth(phi_rec.*width_rec*520000,100),5);
    [r_day,time_day] = accum_last_days(Period,time_rec,smooth(r_rec(2:end),100),5);
	[w_day,time_day] = accum_last_days(Period,t_rec,w_rec,1);
	[h_day,time_day] = accum_last_days(Period,t_rec,h_rec,1);
    evap = Miki_Only_Evap(w_day,h_day,2000);
    Phis{end+1} = phi_day;
    BackPressures{end+1} = r_day;
    WidthRecs{end+1} = w_day;
    HeightRecs{end+1} = h_day;
    Evaporations{end+1} = evap;
    WidthRatios(end+1) = str2num(tmp{2});
    MinWidths(end+1) = str2num(tmp{4});
end
save(filto,'WidthRatios','MinWidths','Phis','BackPressures','WidthRecs','HeightRecs','Evaporations');