function [r_f_k_data, r_stance_kinematics_data,  plate1, plate2, start_frame1, start_frame2, end_frame1, end_frame2] = right_stance_function(sample_freq, analog_freq, down_rate, markers, grw, start, finish, events, output_folder)

% Frame at which events happen in kinematics matrix
try
    right_foot_strike = round((events.Right_Foot_Strike(1)*sample_freq) - start);
    right_foot_off = round((events.Right_Foot_Off(1)*sample_freq) - start);
    
    % If earlier foot-off picked out, then frame will be negative
    if right_foot_off <= right_foot_strike
        right_foot_off = (events.Right_Foot_Off(2)*sample_freq) - start;
        
    end
    
    % Frame at which events happen in force plate matrix size
    r_strike = round(right_foot_strike*down_rate);
    r_off = round(right_foot_off*down_rate);
    
    r_frame = r_off-r_strike+1;
catch
    r_stance_kinematics_data = zeros(finish-start,96);
    r_f_k_data = zeros(finish-start,97);
    start_frame1 = 1;
    start_frame2 = 1;
    end_frame1 = 1;
    end_frame2 = 1;
    plate1 = 'X';
    plate2 = 'X';
    return
end

% Force plate information
COP1  = grw(1).P;
ex_force1 = grw(1).F;
COP2  = grw(2).P;
ex_force2 = grw(2).F;


% Determine which force plate is right/left
if (ex_force1(r_strike-50,1) == 0)&&(ex_force1(r_strike+50,1) ~= 0)
    plate1 = 'R';
    plate2 = 'L';
    r_force(:,1:3) = COP1(r_strike+1:r_strike+r_frame,:);
    r_force(:,4:6) = ex_force1(r_strike+1:r_strike+r_frame,:);
elseif (ex_force2(r_strike-50,1) == 0)&&(ex_force2(r_strike+50,1) ~= 0)
    plate2 = 'R';
    plate1 = 'L';
    r_force(:,1:3) = COP2(r_strike+1:r_strike+r_frame,:);
    r_force(:,4:6) = ex_force2(r_strike+1:r_strike+r_frame,:);
end


% Create shortened matrices (to match dimensions of kinematics data) which
% display a 1 or 0, depending on whether a force is measured

binary_force1 = zeros(finish-start,1);
binary_force2 = zeros(finish-start,1);

for row = 1:(finish-start)
    if ex_force1(row*down_rate,1) ~= 0
        binary_force1(row,1) = 1;
    end
    if ((ex_force1(row*down_rate-1,1) == 1)||(ex_force1(row*down_rate-2,1) == 1)) && ((ex_force1(row*down_rate+1,1) == 1)||(ex_force1(row*down_rate+1,1) == 1))
        binary_force1(row,1) = 1;
    end
    if ex_force2(row*down_rate,1) ~= 0
        binary_force2(row,1) = 1;
    end
    if ((ex_force2(row*down_rate-1,1) == 1)||(ex_force2(row*down_rate-2,1) == 1)) && ((ex_force2(row*down_rate+1,1) == 1)||(ex_force2(row*down_rate+1,1) == 1))
        binary_force2(row,1) = 1;
    end
end


% Find frames with first and last force plate readings

start_frame1 = 1;
while binary_force1(start_frame1,1) == 0
    start_frame1=start_frame1+1;
end

start_frame2 = 1;
while binary_force2(start_frame2,1) == 0
    start_frame2=start_frame2+1;
end

end_frame1 = start_frame1;
try
    while binary_force1(end_frame1,1) == 1
        end_frame1=end_frame1+1;
    end
catch
    r_stance_kinematics_data = zeros(finish-start,96);
    r_f_k_data = zeros(finish-start,97);
    start_frame1 = 1;
    start_frame2 = 1;
    end_frame1 = 1;
    end_frame2 = 1;
    return
end

end_frame2 = start_frame2;
try
    while binary_force2(end_frame2,1) == 1
        end_frame2=end_frame2+1;
    end
catch
    r_stance_kinematics_data = zeros(finish-start,96);
    r_f_k_data = zeros(finish-start,97);
    start_frame1 = 1;
    start_frame2 = 1;
    end_frame1 = 1;
    end_frame2 = 1;
    return
end


try
    RASIS(start:finish,:) = markers.RASIS;
    LASIS(start:finish,:) = markers.LASIS;
    RPSIS(start:finish,:) = markers.RPSIS;
    LPSIS(start:finish,:) = markers.LPSIS;
    RFLE(start:finish,:)= markers.RFLE;
    RFME(start:finish,:) = markers.RFME;
    RT1(start:finish,:) = markers.RT1;
    RT2(start:finish,:) = markers.RT2;
    RT3(start:finish,:) = markers.RT3;
    RFAM(start:finish,:) = markers.RFAM;
    RTAM(start:finish,:) = markers.RTAM;
    RC1(start:finish,:) = markers.RC1;
    RC2(start:finish,:) = markers.RC2;
    RC3(start:finish,:) = markers.RC3;
    RFCC(start:finish,:) = markers.RFCC;
    RFM2(start:finish,:) = markers.RFM2;
    RTF(start:finish,:) = markers.RTF;
    RFMT(start:finish,:) = markers.RFMT;
    LFLE(start:finish,:)= markers.LFLE;
    LFME(start:finish,:) = markers.LFME;
    LT1(start:finish,:) = markers.LT1;
    LT2(start:finish,:) = markers.LT2;
    LT3(start:finish,:) = markers.LT3;
    LFAM(start:finish,:) = markers.LFAM;
    LTAM(start:finish,:) = markers.LTAM;
    LC1(start:finish,:) = markers.LC1;
    LC2(start:finish,:) = markers.LC2;
    LC3(start:finish,:) = markers.LC3;
    LFCC(start:finish,:) = markers.LFCC;
    LFM2(start:finish,:) = markers.LFM2;
    LTF(start:finish,:) = markers.LTF;
    LFMT(start:finish,:) = markers.LFMT;
    
    
    r_stance_kinematics_data = zeros(finish-start,96);
    
    for row = 1:finish-start
        for col = 1:3
            r_stance_kinematics_data(row,col) = RFLE(row+start,col);
            r_stance_kinematics_data(row,col+3) = RFME(row+start,col);
            r_stance_kinematics_data(row,col+6) = RT1(row+start,col);
            r_stance_kinematics_data(row,col+9) = RT2(row+start,col);
            r_stance_kinematics_data(row,col+12) = RT3(row+start,col);
            r_stance_kinematics_data(row,col+15) = RFAM(row+start,col);
            r_stance_kinematics_data(row,col+18) = RTAM(row+start,col);
            r_stance_kinematics_data(row,col+21) = RC1(row+start,col);
            r_stance_kinematics_data(row,col+24) = RC2(row+start,col);
            r_stance_kinematics_data(row,col+27) = RC3(row+start,col);
            r_stance_kinematics_data(row,col+30) = RFCC(row+start,col);
            r_stance_kinematics_data(row,col+33) = RFM2(row+start,col);
            r_stance_kinematics_data(row,col+36) = RTF(row+start,col);
            r_stance_kinematics_data(row,col+39) = RFMT(row+start,col);
            r_stance_kinematics_data(row,col+42) = LFLE(row+start,col);
            r_stance_kinematics_data(row,col+45) = LFME(row+start,col);
            r_stance_kinematics_data(row,col+48) = LT1(row+start,col);
            r_stance_kinematics_data(row,col+51) = LT2(row+start,col);
            r_stance_kinematics_data(row,col+54) = LT3(row+start,col);
            r_stance_kinematics_data(row,col+57) = LFAM(row+start,col);
            r_stance_kinematics_data(row,col+60) = LTAM(row+start,col);
            r_stance_kinematics_data(row,col+63) = LC1(row+start,col);
            r_stance_kinematics_data(row,col+66) = LC2(row+start,col);
            r_stance_kinematics_data(row,col+69) = LC3(row+start,col);
            r_stance_kinematics_data(row,col+72) = LFCC(row+start,col);
            r_stance_kinematics_data(row,col+75) = LFM2(row+start,col);
            r_stance_kinematics_data(row,col+78) = LTF(row+start,col);
            r_stance_kinematics_data(row,col+81) = LFMT(row+start,col);
            r_stance_kinematics_data(row,col+84) = RASIS(row+start,col);
            r_stance_kinematics_data(row,col+87) = LASIS(row+start,col);
            r_stance_kinematics_data(row,col+90) = RPSIS(row+start,col);
            r_stance_kinematics_data(row,col+93) = LPSIS(row+start,col);
        end
    end
    
catch %unlabeled
    r_stance_kinematics_data = zeros(finish-start,96);
    r_f_k_data = zeros(finish-start,97);
    start_frame1 = 1;
    start_frame2 = 1;
    end_frame1 = 1;
    end_frame2 = 1;
    return
end



if plate1 == 'R'
    r_f_k_data = zeros(end_frame1-start_frame1+20, 97);
    for row = 1:end_frame1-start_frame1+20
        r_f_k_data(row,97) = binary_force1(row+start_frame1-11,1);
        for col = 1:96
            r_f_k_data(row,col) = r_stance_kinematics_data(row+start_frame1-11,col);
        end
    end
end


if plate2 == 'R'
    r_f_k_data = zeros(end_frame2-start_frame2+20, 97);
    for row = 1:end_frame2-start_frame2+20
        r_f_k_data(row,97) = binary_force2(row+start_frame2-11,1);
        for col = 1:96
            r_f_k_data(row,col) = r_stance_kinematics_data(row+start_frame2-11,col);
        end
    end
end


