clear;
close all;

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