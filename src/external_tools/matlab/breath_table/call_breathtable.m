function call_breathtable(breath_file)
    % Add path to breathtable definition
    addpath(genpath('/wynton/home/leng/alice-albrecht/matlab_tools/ventilatory_burden'));

    % Load inputs
    inputs = load(breath_file);
    nas_pres = double(inputs.nas_pres);
    fs = inputs.fs;
    opts = inputs.opts;

    % Call your function
    [breath_table, ~] = breathtable(nas_pres, fs, opts);
    colnames = breath_table.Properties.VariableNames;
    breath_array = table2array(breath_table); 

    % Save output
    save(breath_file, 'breath_array', '-append');
end