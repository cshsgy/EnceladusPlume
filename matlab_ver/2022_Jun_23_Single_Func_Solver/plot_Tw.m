id = 10;
temp = [];
for i=1:length(time_rec)
	temp(end+1) = T_rec{i}(id,1);
end
Ps = round(max(time_rec/Period));
plot(time_rec/Period,smooth(temp,20),'linewidth',2);
xlim([Ps-1 Ps]);
xlabel('Time (Day)');
ylabel(sprintf('Temp (K) at %.1f meters above neutral',z_wet(id)));
grid on;
saveas(gcf,'temp_10.png');