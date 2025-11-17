function get_Farhad_Kenn_rpeaks(ecg_file)
    % Add the path to the R-peak detection code

    addpath("/wynton/home/leng/alice-albrecht/PSG_Pipeline/src/external_tools/matlab/mros_rpeaks_detection/UCSF_HRV_Tool_Matlab_GUI_Kenn_Farhad");

    % Load the subject-specific ECG file (must contain ecg, fs)
    load(ecg_file);

    % Detect R peaks and artifacts
    try
        [R_Index, Art_Index] = DetectR_Art(ecg, double(fs));
    catch
        R_Index = [];
        Art_Index = [];
        warning('No R-peaks detected for this subject.');
    end

    % Save results back to the same file (append)
    save(ecg_file, 'R_Index', 'Art_Index', '-append');
end