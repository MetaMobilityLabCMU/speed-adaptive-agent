function processFolder(inputFolder, outputFolder)
    % Get the list of all files and subfolders
    entries = dir(inputFolder);

    % Iterate through each entry
    for i = 1:length(entries)
        entry = entries(i);

        % Skip '.' and '..' entries
        if strcmp(entry.name, '.') || strcmp(entry.name, '..')
            continue;
        end

        % Full paths for input and output
        inputPath = fullfile(inputFolder, entry.name);
        outputPath = fullfile(outputFolder, entry.name);

        if entry.isdir
            % Create corresponding subfolder in the output
            if ~exist(outputPath, 'dir')
                mkdir(outputPath);
            end
            % Recursively process the subfolder
            processFolder(inputPath, outputPath);
        elseif endsWith(entry.name, '.mat')
            % Load the .mat file
            matData = load(inputPath);

            % Get the variable names
            varNames = fieldnames(matData);

            % Save each variable as a CSV file
            for j = 1:numel(varNames)
                varData = matData.(varNames{j});

                % Ensure the variable is numeric or can be written as CSV
                if isnumeric(varData) || islogical(varData)
                    csvFileName = fullfile(outputFolder, replace(entry.name, '.mat', ['_' varNames{j} '.csv']));
                    writematrix(varData, csvFileName);
                else
                    warning('Skipping non-numeric variable "%s" in file "%s".', varNames{j}, entry.name);
                end
            end
        end
    end
end


gatechFolder = "../data/CAMARGO_ET_AL_J_BIOMECH_DATASET";
outputFolder = fullfile(gatechFolder, 'dataset_csv');

% Ensure the output folder exists
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

% Recursively process the root folder
processFolder(gatechFolder, outputFolder);




clear;
close all;
gatech_folder = "../data/CAMARGO_ET_AL_J_BIOMECH_DATASET";
activity = "treadmill";
flags = ["ik"; "gcRight"; "conditions"];
folder_name = "levelground_test";

sub_folders = ["ik"; "id"; "conditions"];

for k=1:length(sub_folders)
    mat_path = strcat(folder_name, '/', 'mat', '/', sub_folders(k));
    csv_path = strcat(folder_name, '/', 'csv', '/', sub_folders(k));

    mat_fnames = dir(mat_path);

    for j=1:length(mat_fnames)
        mat_fname = strcat(mat_path, '/', mat_fnames(j).name);

        if contains(mat_fname,".mat")
            csv_fname = strcat(csv_path, '/', strcat(extractBetween(mat_fnames(j).name, 1, strlength(mat_fnames(j).name)-3), 'csv'));
        
            temp = load(mat_fname);
            disp(' ');
            disp(mat_fname);
            disp(csv_fname);
            if contains(mat_fname,"conditions")
                writetable(temp.labels, csv_fname);
            else
                writetable(temp.data, csv_fname);
            end
        end
    end
end