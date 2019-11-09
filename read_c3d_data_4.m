clearvars

% Input file information
input_folder = 'UROP\OA Study';

% Output file information
output_folder = 'UROP\Processed Force and Kinematics Data';


% Search through database for all walking c3d files to process

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
                cd(sprintf('\\%s', output_folder));
                mkdir(sprintf('\\%s\\%s', output_folder, subject));
                cd(sprintf('\\%s\\%s\\%s', input_folder, date, subject));
                all_sessions = dir;
                
                for k = 3:numel(all_sessions)
                    if all_sessions(k,1).isdir == 1
                        session = all_sessions(k,1).name;
                        cd(sprintf('\\%s\\%s\\%s\\%s', input_folder, date, subject, all_sessions(k,1).name));
                        all_walking = dir;
                        
                        for m = 3:length(all_walking)
                            cd(sprintf('\\%s\\%s\\%s\\%s', input_folder, date, subject, all_sessions(k,1).name));
                            if isfile(sprintf('\\%s\\%s\\%s\\%s\\walking %s.c3d', input_folder, date, subject, session, m));
                                load_file = sprintf('\\%s\\%s\\%s\\%s\\walking %s.c3d', input_folder, date, subject, session, m);
                                btk_load = [load_file];
                                
                                acq = btkReadAcquisition(btk_load);
                                
                                
                                % Sample frequency (motion tracking vicon)
                                sample_freq = btkGetPointFrequency(acq);
                                
                                
                                % Analog Frequency (forceplates)
                                analog_freq = btkGetAnalogFrequency(acq);
                                
                                % Downsampling Rate
                                down_rate = analog_freq/sample_freq;
                                
                                markers = btkGetMarkers(acq);
                                grw = btkGetGroundReactionWrenches(acq);
                                
                                
                                start = btkGetFirstFrame(acq);      % These use relative frame numbering. So if you want Nexus frame # 200, you need to go to "200 - start + 1"
                                finish = btkGetLastFrame(acq);      % When event markers are set up, use them to define the start/end of the data
                                
                                % Events from the trial
                                [events] = btkGetEvents(acq);
                                
                                
                                try
                                [r_f_k_data, r_stance_kinematics_data, plate1, plate2, start_frame1, start_frame2, end_frame1, end_frame2] = right_stance_function(sample_freq, analog_freq, down_rate, markers, grw, start, finish, events, output_folder);
                                if r_stance_kinematics_data ~= zeros(finish-start,96)
                                    
                                    % Output right stance data to file in output folder
                                    
                                    cd ..
                                    cd ..
                                    cd ..
                                    cd ..
                                    cd ..
                                    cd(sprintf('%s\\%s', output_folder, subject));
                                    filename = sprintf('r_%s', m);
                                    fid = fopen(filename, 'wt');
                                    
                                    if plate1 == 'R'
                                        fprintf(fid, 'RFLE(x)\t');
                                        fprintf(fid, 'RFLE(y)\t');
                                        fprintf(fid, 'RFLE(z)\t');
                                        fprintf(fid, 'RFME(x)\t');
                                        fprintf(fid, 'RFME(y)\t');
                                        fprintf(fid, 'RFME(z)\t');
                                        fprintf(fid, 'RT1(x)\t');
                                        fprintf(fid, 'RT1(y)\t');
                                        fprintf(fid, 'RT1(z)\t');
                                        fprintf(fid, 'RT2(x)\t');
                                        fprintf(fid, 'RT2(y)\t');
                                        fprintf(fid, 'RT2(z)\t');
                                        fprintf(fid, 'RT3(x)\t');
                                        fprintf(fid, 'RT3(y)\t');
                                        fprintf(fid, 'RT3(z)\t');
                                        fprintf(fid, 'RFAM(x)\t');
                                        fprintf(fid, 'RFAM(y)\t');
                                        fprintf(fid, 'RFAM(z)\t');
                                        fprintf(fid, 'RTAM(x)\t');
                                        fprintf(fid, 'RTAM(y)\t');
                                        fprintf(fid, 'RTAM(z)\t');
                                        fprintf(fid, 'RC1(x)\t');
                                        fprintf(fid, 'RC1(y)\t');
                                        fprintf(fid, 'RC1(z)\t');
                                        fprintf(fid, 'RC2(x)\t');
                                        fprintf(fid, 'RC2(y)\t');
                                        fprintf(fid, 'RC2(z)\t');
                                        fprintf(fid, 'RC3(x)\t');
                                        fprintf(fid, 'RC3(y)\t');
                                        fprintf(fid, 'RC3(z)\t');
                                        fprintf(fid, 'RFCC(x)\t');
                                        fprintf(fid, 'RFCC(y)\t');
                                        fprintf(fid, 'RFCC(z)\t');
                                        fprintf(fid, 'RFM2(x)\t');
                                        fprintf(fid, 'RFM2(y)\t');
                                        fprintf(fid, 'RFM2(z)\t');
                                        fprintf(fid, 'RTF(x)\t');
                                        fprintf(fid, 'RTF(y)\t');
                                        fprintf(fid, 'RTF(z)\t');
                                        fprintf(fid, 'RFMT(x)\t');
                                        fprintf(fid, 'RFMT(y)\t');
                                        fprintf(fid, 'RFMT(z)\t');
                                        fprintf(fid, 'LFLE(x)\t');
                                        fprintf(fid, 'LFLE(y)\t');
                                        fprintf(fid, 'LFLE(z)\t');
                                        fprintf(fid, 'LFME(x)\t');
                                        fprintf(fid, 'LFME(y)\t');
                                        fprintf(fid, 'LFME(z)\t');
                                        fprintf(fid, 'LT1(x)\t');
                                        fprintf(fid, 'LT1(y)\t');
                                        fprintf(fid, 'LT1(z)\t');
                                        fprintf(fid, 'LT2(x)\t');
                                        fprintf(fid, 'LT2(y)\t');
                                        fprintf(fid, 'LT2(z)\t');
                                        fprintf(fid, 'LT3(x)\t');
                                        fprintf(fid, 'LT3(y)\t');
                                        fprintf(fid, 'LT3(z)\t');
                                        fprintf(fid, 'LFAM(x)\t');
                                        fprintf(fid, 'LFAM(y)\t');
                                        fprintf(fid, 'LFAM(z)\t');
                                        fprintf(fid, 'LTAM(x)\t');
                                        fprintf(fid, 'LTAM(y)\t');
                                        fprintf(fid, 'LTAM(z)\t');
                                        fprintf(fid, 'LC1(x)\t');
                                        fprintf(fid, 'LC1(y)\t');
                                        fprintf(fid, 'LC1(z)\t');
                                        fprintf(fid, 'LC2(x)\t');
                                        fprintf(fid, 'LC2(y)\t');
                                        fprintf(fid, 'LC2(z)\t');
                                        fprintf(fid, 'LC3(x)\t');
                                        fprintf(fid, 'LC3(y)\t');
                                        fprintf(fid, 'LC3(z)\t');
                                        fprintf(fid, 'LFCC(x)\t');
                                        fprintf(fid, 'LFCC(y)\t');
                                        fprintf(fid, 'LFCC(z)\t');
                                        fprintf(fid, 'LFM2(x)\t');
                                        fprintf(fid, 'LFM2(y)\t');
                                        fprintf(fid, 'LFM2(z)\t');
                                        fprintf(fid, 'LTF(x)\t');
                                        fprintf(fid, 'LTF(y)\t');
                                        fprintf(fid, 'LTF(z)\t');
                                        fprintf(fid, 'LFMT(x)\t');
                                        fprintf(fid, 'LFMT(y)\t');
                                        fprintf(fid, 'LFMT(z)\t');
                                        fprintf(fid, 'RASIS(x)\t');
                                        fprintf(fid, 'RASIS(y)\t');
                                        fprintf(fid, 'RASIS(z)\t');
                                        fprintf(fid, 'LASIS(x)\t');
                                        fprintf(fid, 'LASIS(y)\t');
                                        fprintf(fid, 'LASIS(z)\t');
                                        fprintf(fid, 'RPSIS(x)\t');
                                        fprintf(fid, 'RPSIS(y)\t');
                                        fprintf(fid, 'RPSIS(z)\t');
                                        fprintf(fid, 'LPSIS(x)\t');
                                        fprintf(fid, 'LPSIS(y)\t');
                                        fprintf(fid, 'LPSIS(z)\t');
                                        fprintf(fid, 'FORCE PLATE\n');
                                        for row = 1:end_frame1-start_frame1+20
                                            fprintf(fid,'%4.2f\t', r_f_k_data(row,:));
                                            fprintf(fid, '\n');
                                        end
                                        
                                        fclose(fid);
                                    end
                                    
                                    if plate2 == 'R'
                                        fprintf(fid, 'RFLE(x)\t');
                                        fprintf(fid, 'RFLE(y)\t');
                                        fprintf(fid, 'RFLE(z)\t');
                                        fprintf(fid, 'RFME(x)\t');
                                        fprintf(fid, 'RFME(y)\t');
                                        fprintf(fid, 'RFME(z)\t');
                                        fprintf(fid, 'RT1(x)\t');
                                        fprintf(fid, 'RT1(y)\t');
                                        fprintf(fid, 'RT1(z)\t');
                                        fprintf(fid, 'RT2(x)\t');
                                        fprintf(fid, 'RT2(y)\t');
                                        fprintf(fid, 'RT2(z)\t');
                                        fprintf(fid, 'RT3(x)\t');
                                        fprintf(fid, 'RT3(y)\t');
                                        fprintf(fid, 'RT3(z)\t');
                                        fprintf(fid, 'RFAM(x)\t');
                                        fprintf(fid, 'RFAM(y)\t');
                                        fprintf(fid, 'RFAM(z)\t');
                                        fprintf(fid, 'RTAM(x)\t');
                                        fprintf(fid, 'RTAM(y)\t');
                                        fprintf(fid, 'RTAM(z)\t');
                                        fprintf(fid, 'RC1(x)\t');
                                        fprintf(fid, 'RC1(y)\t');
                                        fprintf(fid, 'RC1(z)\t');
                                        fprintf(fid, 'RC2(x)\t');
                                        fprintf(fid, 'RC2(y)\t');
                                        fprintf(fid, 'RC2(z)\t');
                                        fprintf(fid, 'RC3(x)\t');
                                        fprintf(fid, 'RC3(y)\t');
                                        fprintf(fid, 'RC3(z)\t');
                                        fprintf(fid, 'RFCC(x)\t');
                                        fprintf(fid, 'RFCC(y)\t');
                                        fprintf(fid, 'RFCC(z)\t');
                                        fprintf(fid, 'RFM2(x)\t');
                                        fprintf(fid, 'RFM2(y)\t');
                                        fprintf(fid, 'RFM2(z)\t');
                                        fprintf(fid, 'RTF(x)\t');
                                        fprintf(fid, 'RTF(y)\t');
                                        fprintf(fid, 'RTF(z)\t');
                                        fprintf(fid, 'RFMT(x)\t');
                                        fprintf(fid, 'RFMT(y)\t');
                                        fprintf(fid, 'RFMT(z)\t');
                                        fprintf(fid, 'LFLE(x)\t');
                                        fprintf(fid, 'LFLE(y)\t');
                                        fprintf(fid, 'LFLE(z)\t');
                                        fprintf(fid, 'LFME(x)\t');
                                        fprintf(fid, 'LFME(y)\t');
                                        fprintf(fid, 'LFME(z)\t');
                                        fprintf(fid, 'LT1(x)\t');
                                        fprintf(fid, 'LT1(y)\t');
                                        fprintf(fid, 'LT1(z)\t');
                                        fprintf(fid, 'LT2(x)\t');
                                        fprintf(fid, 'LT2(y)\t');
                                        fprintf(fid, 'LT2(z)\t');
                                        fprintf(fid, 'LT3(x)\t');
                                        fprintf(fid, 'LT3(y)\t');
                                        fprintf(fid, 'LT3(z)\t');
                                        fprintf(fid, 'LFAM(x)\t');
                                        fprintf(fid, 'LFAM(y)\t');
                                        fprintf(fid, 'LFAM(z)\t');
                                        fprintf(fid, 'LTAM(x)\t');
                                        fprintf(fid, 'LTAM(y)\t');
                                        fprintf(fid, 'LTAM(z)\t');
                                        fprintf(fid, 'LC1(x)\t');
                                        fprintf(fid, 'LC1(y)\t');
                                        fprintf(fid, 'LC1(z)\t');
                                        fprintf(fid, 'LC2(x)\t');
                                        fprintf(fid, 'LC2(y)\t');
                                        fprintf(fid, 'LC2(z)\t');
                                        fprintf(fid, 'LC3(x)\t');
                                        fprintf(fid, 'LC3(y)\t');
                                        fprintf(fid, 'LC3(z)\t');
                                        fprintf(fid, 'LFCC(x)\t');
                                        fprintf(fid, 'LFCC(y)\t');
                                        fprintf(fid, 'LFCC(z)\t');
                                        fprintf(fid, 'LFM2(x)\t');
                                        fprintf(fid, 'LFM2(y)\t');
                                        fprintf(fid, 'LFM2(z)\t');
                                        fprintf(fid, 'LTF(x)\t');
                                        fprintf(fid, 'LTF(y)\t');
                                        fprintf(fid, 'LTF(z)\t');
                                        fprintf(fid, 'LFMT(x)\t');
                                        fprintf(fid, 'LFMT(y)\t');
                                        fprintf(fid, 'LFMT(z)\t');
                                        fprintf(fid, 'RASIS(x)\t');
                                        fprintf(fid, 'RASIS(y)\t');
                                        fprintf(fid, 'RASIS(z)\t');
                                        fprintf(fid, 'LASIS(x)\t');
                                        fprintf(fid, 'LASIS(y)\t');
                                        fprintf(fid, 'LASIS(z)\t');
                                        fprintf(fid, 'RPSIS(x)\t');
                                        fprintf(fid, 'RPSIS(y)\t');
                                        fprintf(fid, 'RPSIS(z)\t');
                                        fprintf(fid, 'LPSIS(x)\t');
                                        fprintf(fid, 'LPSIS(y)\t');
                                        fprintf(fid, 'LPSIS(z)\t');
                                        fprintf(fid, 'FORCE PLATE\n');
                                        for row = 1:end_frame2-start_frame2+20
                                            fprintf(fid,'%4.2f\t', r_f_k_data(row,:));
                                            fprintf(fid, '\n');
                                        end
                                        
                                        fclose(fid);
                                    end
                                end
                                
                                catch
                                    continue
                                end
                                
                                
                                try
                                    [l_f_k_data, l_stance_kinematics_data, plate1, plate2, start_frame1, start_frame2, end_frame1, end_frame2] = left_stance_function(sample_freq, analog_freq, down_rate, markers, grw, start, finish, events, output_folder);
                                    if l_stance_kinematics_data ~= zeros(finish-start,96)
                                        
                                        % Output left stance data to file in output folder
                                        
                                        cd ..
                                        cd ..
                                        cd ..
                                        cd ..
                                        cd ..
                                        cd(sprintf('%s\\%s', output_folder, subject));
                                        filename = sprintf('l_%s', m);
                                        fid = fopen(filename, 'wt');
                                        
                                        if plate1 == 'L'
                                            fprintf(fid, 'LFLE(x)\t');
                                            fprintf(fid, 'LFLE(y)\t');
                                            fprintf(fid, 'LFLE(z)\t');
                                            fprintf(fid, 'LFME(x)\t');
                                            fprintf(fid, 'LFME(y)\t');
                                            fprintf(fid, 'LFME(z)\t');
                                            fprintf(fid, 'LT1(x)\t');
                                            fprintf(fid, 'LT1(y)\t');
                                            fprintf(fid, 'LT1(z)\t');
                                            fprintf(fid, 'LT2(x)\t');
                                            fprintf(fid, 'LT2(y)\t');
                                            fprintf(fid, 'LT2(z)\t');
                                            fprintf(fid, 'LT3(x)\t');
                                            fprintf(fid, 'LT3(y)\t');
                                            fprintf(fid, 'LT3(z)\t');
                                            fprintf(fid, 'LFAM(x)\t');
                                            fprintf(fid, 'LFAM(y)\t');
                                            fprintf(fid, 'LFAM(z)\t');
                                            fprintf(fid, 'LTAM(x)\t');
                                            fprintf(fid, 'LTAM(y)\t');
                                            fprintf(fid, 'LTAM(z)\t');
                                            fprintf(fid, 'LC1(x)\t');
                                            fprintf(fid, 'LC1(y)\t');
                                            fprintf(fid, 'LC1(z)\t');
                                            fprintf(fid, 'LC2(x)\t');
                                            fprintf(fid, 'LC2(y)\t');
                                            fprintf(fid, 'LC2(z)\t');
                                            fprintf(fid, 'LC3(x)\t');
                                            fprintf(fid, 'LC3(y)\t');
                                            fprintf(fid, 'LC3(z)\t');
                                            fprintf(fid, 'LFCC(x)\t');
                                            fprintf(fid, 'LFCC(y)\t');
                                            fprintf(fid, 'LFCC(z)\t');
                                            fprintf(fid, 'LFM2(x)\t');
                                            fprintf(fid, 'LFM2(y)\t');
                                            fprintf(fid, 'LFM2(z)\t');
                                            fprintf(fid, 'LTF(x)\t');
                                            fprintf(fid, 'LTF(y)\t');
                                            fprintf(fid, 'LTF(z)\t');
                                            fprintf(fid, 'LFMT(x)\t');
                                            fprintf(fid, 'LFMT(y)\t');
                                            fprintf(fid, 'LFMT(z)\t');
                                            fprintf(fid, 'RFLE(x)\t');
                                            fprintf(fid, 'RFLE(y)\t');
                                            fprintf(fid, 'RFLE(z)\t');
                                            fprintf(fid, 'RFME(x)\t');
                                            fprintf(fid, 'RFME(y)\t');
                                            fprintf(fid, 'RFME(z)\t');
                                            fprintf(fid, 'RT1(x)\t');
                                            fprintf(fid, 'RT1(y)\t');
                                            fprintf(fid, 'RT1(z)\t');
                                            fprintf(fid, 'RT2(x)\t');
                                            fprintf(fid, 'RT2(y)\t');
                                            fprintf(fid, 'RT2(z)\t');
                                            fprintf(fid, 'RT3(x)\t');
                                            fprintf(fid, 'RT3(y)\t');
                                            fprintf(fid, 'RT3(z)\t');
                                            fprintf(fid, 'RFAM(x)\t');
                                            fprintf(fid, 'RFAM(y)\t');
                                            fprintf(fid, 'RFAM(z)\t');
                                            fprintf(fid, 'RTAM(x)\t');
                                            fprintf(fid, 'RTAM(y)\t');
                                            fprintf(fid, 'RTAM(z)\t');
                                            fprintf(fid, 'RC1(x)\t');
                                            fprintf(fid, 'RC1(y)\t');
                                            fprintf(fid, 'RC1(z)\t');
                                            fprintf(fid, 'RC2(x)\t');
                                            fprintf(fid, 'RC2(y)\t');
                                            fprintf(fid, 'RC2(z)\t');
                                            fprintf(fid, 'RC3(x)\t');
                                            fprintf(fid, 'RC3(y)\t');
                                            fprintf(fid, 'RC3(z)\t');
                                            fprintf(fid, 'RFCC(x)\t');
                                            fprintf(fid, 'RFCC(y)\t');
                                            fprintf(fid, 'RFCC(z)\t');
                                            fprintf(fid, 'RFM2(x)\t');
                                            fprintf(fid, 'RFM2(y)\t');
                                            fprintf(fid, 'RFM2(z)\t');
                                            fprintf(fid, 'RTF(x)\t');
                                            fprintf(fid, 'RTF(y)\t');
                                            fprintf(fid, 'RTF(z)\t');
                                            fprintf(fid, 'RFMT(x)\t');
                                            fprintf(fid, 'RFMT(y)\t');
                                            fprintf(fid, 'RFMT(z)\t');
                                            fprintf(fid, 'RASIS(x)\t');
                                            fprintf(fid, 'RASIS(y)\t');
                                            fprintf(fid, 'RASIS(z)\t');
                                            fprintf(fid, 'LASIS(x)\t');
                                            fprintf(fid, 'LASIS(y)\t');
                                            fprintf(fid, 'LASIS(z)\t');
                                            fprintf(fid, 'RPSIS(x)\t');
                                            fprintf(fid, 'RPSIS(y)\t');
                                            fprintf(fid, 'RPSIS(z)\t');
                                            fprintf(fid, 'LPSIS(x)\t');
                                            fprintf(fid, 'LPSIS(y)\t');
                                            fprintf(fid, 'LPSIS(z)\t');
                                            fprintf(fid, 'FORCE PLATE\n');
                                            for row = 1:end_frame1-start_frame1+20
                                                fprintf(fid,'%4.2f\t', l_f_k_data(row,:));
                                                fprintf(fid, '\n');
                                            end
                                        end
                                        
                                        if plate2 == 'L'
                                            fprintf(fid, 'LFLE(x)\t');
                                            fprintf(fid, 'LFLE(y)\t');
                                            fprintf(fid, 'LFLE(z)\t');
                                            fprintf(fid, 'LFME(x)\t');
                                            fprintf(fid, 'LFME(y)\t');
                                            fprintf(fid, 'LFME(z)\t');
                                            fprintf(fid, 'LT1(x)\t');
                                            fprintf(fid, 'LT1(y)\t');
                                            fprintf(fid, 'LT1(z)\t');
                                            fprintf(fid, 'LT2(x)\t');
                                            fprintf(fid, 'LT2(y)\t');
                                            fprintf(fid, 'LT2(z)\t');
                                            fprintf(fid, 'LT3(x)\t');
                                            fprintf(fid, 'LT3(y)\t');
                                            fprintf(fid, 'LT3(z)\t');
                                            fprintf(fid, 'LFAM(x)\t');
                                            fprintf(fid, 'LFAM(y)\t');
                                            fprintf(fid, 'LFAM(z)\t');
                                            fprintf(fid, 'LTAM(x)\t');
                                            fprintf(fid, 'LTAM(y)\t');
                                            fprintf(fid, 'LTAM(z)\t');
                                            fprintf(fid, 'LC1(x)\t');
                                            fprintf(fid, 'LC1(y)\t');
                                            fprintf(fid, 'LC1(z)\t');
                                            fprintf(fid, 'LC2(x)\t');
                                            fprintf(fid, 'LC2(y)\t');
                                            fprintf(fid, 'LC2(z)\t');
                                            fprintf(fid, 'LC3(x)\t');
                                            fprintf(fid, 'LC3(y)\t');
                                            fprintf(fid, 'LC3(z)\t');
                                            fprintf(fid, 'LFCC(x)\t');
                                            fprintf(fid, 'LFCC(y)\t');
                                            fprintf(fid, 'LFCC(z)\t');
                                            fprintf(fid, 'LFM2(x)\t');
                                            fprintf(fid, 'LFM2(y)\t');
                                            fprintf(fid, 'LFM2(z)\t');
                                            fprintf(fid, 'LTF(x)\t');
                                            fprintf(fid, 'LTF(y)\t');
                                            fprintf(fid, 'LTF(z)\t');
                                            fprintf(fid, 'LFMT(x)\t');
                                            fprintf(fid, 'LFMT(y)\t');
                                            fprintf(fid, 'LFMT(z)\t');
                                            fprintf(fid, 'RFLE(x)\t');
                                            fprintf(fid, 'RFLE(y)\t');
                                            fprintf(fid, 'RFLE(z)\t');
                                            fprintf(fid, 'RFME(x)\t');
                                            fprintf(fid, 'RFME(y)\t');
                                            fprintf(fid, 'RFME(z)\t');
                                            fprintf(fid, 'RT1(x)\t');
                                            fprintf(fid, 'RT1(y)\t');
                                            fprintf(fid, 'RT1(z)\t');
                                            fprintf(fid, 'RT2(x)\t');
                                            fprintf(fid, 'RT2(y)\t');
                                            fprintf(fid, 'RT2(z)\t');
                                            fprintf(fid, 'RT3(x)\t');
                                            fprintf(fid, 'RT3(y)\t');
                                            fprintf(fid, 'RT3(z)\t');
                                            fprintf(fid, 'RFAM(x)\t');
                                            fprintf(fid, 'RFAM(y)\t');
                                            fprintf(fid, 'RFAM(z)\t');
                                            fprintf(fid, 'RTAM(x)\t');
                                            fprintf(fid, 'RTAM(y)\t');
                                            fprintf(fid, 'RTAM(z)\t');
                                            fprintf(fid, 'RC1(x)\t');
                                            fprintf(fid, 'RC1(y)\t');
                                            fprintf(fid, 'RC1(z)\t');
                                            fprintf(fid, 'RC2(x)\t');
                                            fprintf(fid, 'RC2(y)\t');
                                            fprintf(fid, 'RC2(z)\t');
                                            fprintf(fid, 'RC3(x)\t');
                                            fprintf(fid, 'RC3(y)\t');
                                            fprintf(fid, 'RC3(z)\t');
                                            fprintf(fid, 'RFCC(x)\t');
                                            fprintf(fid, 'RFCC(y)\t');
                                            fprintf(fid, 'RFCC(z)\t');
                                            fprintf(fid, 'RFM2(x)\t');
                                            fprintf(fid, 'RFM2(y)\t');
                                            fprintf(fid, 'RFM2(z)\t');
                                            fprintf(fid, 'RTF(x)\t');
                                            fprintf(fid, 'RTF(y)\t');
                                            fprintf(fid, 'RTF(z)\t');
                                            fprintf(fid, 'RFMT(x)\t');
                                            fprintf(fid, 'RFMT(y)\t');
                                            fprintf(fid, 'RFMT(z)\t');
                                            fprintf(fid, 'RASIS(x)\t');
                                            fprintf(fid, 'RASIS(y)\t');
                                            fprintf(fid, 'RASIS(z)\t');
                                            fprintf(fid, 'LASIS(x)\t');
                                            fprintf(fid, 'LASIS(y)\t');
                                            fprintf(fid, 'LASIS(z)\t');
                                            fprintf(fid, 'RPSIS(x)\t');
                                            fprintf(fid, 'RPSIS(y)\t');
                                            fprintf(fid, 'RPSIS(z)\t');
                                            fprintf(fid, 'LPSIS(x)\t');
                                            fprintf(fid, 'LPSIS(y)\t');
                                            fprintf(fid, 'LPSIS(z)\t');
                                            fprintf(fid, 'FORCE PLATE\n');
                                            for row = 1:end_frame2-start_frame2+20
                                                fprintf(fid,'%4.2f\t', l_f_k_data(row,:));
                                                fprintf(fid, '\n');
                                            end
                                        end
                                        fclose(fid);
                                    end
                                catch
                                    continue
                                end
                                
                                btkDeleteAcquisition(acq);
                                
                            end
                        end
                    end
                end
            end
        end
    end
end

