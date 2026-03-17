folder_name = '/home/shc/Enceladus/2021_25Jul_ParallelCrackDyn/MATLAB_General_Solver_Corrected/';
fils = dir(folder_name);
parfor i=1:length(fils)
    if ~contains(fils(i).name,'.mat')
		continue
    end
    main_func(folder_name,fils(i).name);
end
