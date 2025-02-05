% Recursively convert all .mat files in a folder to .csv files 

clear;

folderPath = '../data/CAMARGO_ET_AL_J_BIOMECH_DATASET';
folders = dir(folderPath);
folders = folders([folders.isdir]); % Keep only directories
subjects = string({folders.name}); % Convert to string array
subjects = subjects(~ismember(subjects, [".", ".."])); % Exclude '.' and '..'

activity = 'treadmill';
features = ["ik"; "conditions"; "gcRight"];

mkdir(fullfile('../data/', 'dataset_csv'));

for i=1:length(subjects)
    mkdir(fullfile('../data/dataset_csv', subjects(i)));
    csvPath = strcat('../data/dataset_csv/', subjects(i));
    mkdir(fullfile(csvPath, activity));
    csvPath = strcat(csvPath, '/', activity);

    subFolderPath = strcat(folderPath, '/', subjects(i));
    items = dir(subFolderPath);
    items = items([items.isdir]); 
    dateFolder = string({items.name}); 
    dateFolder = dateFolder(~startsWith(dateFolder, ".") & dateFolder ~= "osimxml");
    
    subjectPath = strcat(subFolderPath, '/', dateFolder, '/', activity);

    for j=1:length(features)
        mkdir(fullfile(csvPath, features(j)));
        csvFeaturePath = strcat(csvPath, '/', features(j));
        matFeaturePath = strcat(subjectPath, '/', features(j));

        mat_fnames = dir(matFeaturePath);
        for k=1:length(mat_fnames)
            mat_fname = strcat(matFeaturePath, '/', mat_fnames(k).name);
    
            if contains(mat_fname,".mat")
                csv_fname = strcat(csvFeaturePath, '/', strcat(extractBetween(mat_fnames(k).name, 1, strlength(mat_fnames(k).name)-3), 'csv'));
    
                temp = load(mat_fname);
                disp(' ');
                disp(mat_fname);
                disp(csv_fname);
                if contains(mat_fname,"conditions")
                    writetable(temp.speed, csv_fname);
                else
                    writetable(temp.data, csv_fname);
                end
            end
        end
    end
end