clearvars

% Output file information
output_folder = 'UROP\Processed F K Data';

% Input file information
input_folder = 'UROP\OA Study';
cd(sprintf('\\%s', input_folder))

all_dates = dir;
datessize=length(all_dates);


for i = 3:numel(all_dates)
    if (all_dates(i,1).isdir == 1)
        date = all_dates(i,1).name;
        cd(sprintf('\\%s\\%s', input_folder, date));
        all_subjects = dir;
        
        for j = 3:numel(all_subjects)
            if (all_subjects(j,1).isdir == 1)
                subject = all_subjects(j,1).name;
                cd(sprintf('\\%s\\%s\\%s', input_folder, date, subject));
                all_sessions = dir;
                
                for k = 3:numel(all_sessions)
                    if all_sessions(k,1).isdir == 1
                        session = all_sessions(k,1).name;
                        cd(sprintf('\\%s\\%s\\%s\\%s', input_folder, date, subject, all_sessions(k,1).name));
                        all_walking = dir;
                        
                        for m = 3:length(all_walking)
                            if isfile(sprintf('\\%s\\%s\\%s\\%s\\walking %s.c3d', input_folder, date, subject, session, m));
                                load_file = sprintf('\\%s\\%s\\%s\\%s\\walking %s.c3d', input_folder, date, subject, session, m);
                                btk_load = [load_file];
                            end                           
                        end
                    end
                end
            end
        end
    end
end