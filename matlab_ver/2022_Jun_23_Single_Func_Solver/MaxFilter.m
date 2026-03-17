function new_arr = MaxFilter(arr,half_size)
new_arr = arr;
for i=1:length(arr)
	new_arr(i) = max(arr(max(i-half_size,1):min(length(arr),i+half_size)));
end
end