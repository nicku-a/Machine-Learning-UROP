function [l_f_k_data, l_stance_kinematics_data,  plate1, plate2, start_frame1, start_frame2, end_frame1, end_frame2] = left_stance_function(sample_freq, analog_freq, down_rate, markers, grw, start, finish, events, output_folder)

% Frame at which events happen in kinematics matrix
try
    left_foot_strike = round((events.Left_Foot_Strike(1)*sample_freq) - start);
    left_foot_off = round((events.Left_Foot_Off(1)*sample_freq) - start);
    
    % If earlier foot-off picked out, then frame will be negative
    if left_foot_off <= left_foot_strike
        left_foot_off = (events.Left_Foot_Off(2)*sample_freq) - start;
        
    end
    
    % Frame at which events happen in force plate matrix size
    l_strike = round(left_foot_strike*down_rate);
    l_off = round(left_foot_off*down_rate);
    
    l_frame = l_off-l_strike+1;
catch
    l_stance_kinematics_data = zeros(finish-start,96);
    l_f_k_data = zeros(finish-start,97);
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
if (ex_force1(l_strike-50,1) == 0)&&(ex_force1(l_strike+50,1) ~= 0)
    plate1 = 'L';
    plate2 = 'R';
    l_force(:,1:3) = COP1(l_strike+1:l_strike+l_frame,:);
    l_force(:,4:6) = ex_force1(l_strike+1:l_strike+l_frame,:);
elseif (ex_force2(l_strike-50,1) == 0)&&(ex_force2(l_strike+50,1) ~= 0)
    plate2 = 'L';
    plate1 = 'R';
    l_force(:,1:3) = COP2(l_strike+1:l_strike+l_frame,:);
    l_force(:,4:6) = ex_force2(l_strike+1:l_strike+l_frame,:);
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

try
    end_frame1 = start_frame1;
    while binary_force1(end_frame1,1) == 1
        end_frame1=end_frame1+1;
    end
catch
    l_stance_kinematics_data = zeros(finish-start,96);
    l_f_k_data = zeros(finish-start,97);
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
    l_stance_kinematics_data = zeros(finish-start,96);
    l_f_k_data = zeros(finish-start,97);
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
    
    
    l_stance_kinematics_data = zeros(finish-start,96);
    
    for row = 1:finish-start
        for col = 1:3
            l_stance_kinematics_data(row,col) = LFLE(row+start,col);
            l_stance_kinematics_data(row,col+3) = LFME(row+start,col);
            l_stance_kinematics_data(row,col+6) = LT1(row+start,col);
            l_stance_kinematics_data(row,col+9) = LT2(row+start,col);
            l_stance_kinematics_data(row,col+12) = LT3(row+start,col);
            l_stance_kinematics_data(row,col+15) = LFAM(row+start,col);
            l_stance_kinematics_data(row,col+18) = LTAM(row+start,col);
            l_stance_kinematics_data(row,col+21) = LC1(row+start,col);
            l_stance_kinematics_data(row,col+24) = LC2(row+start,col);
            l_stance_kinematics_data(row,col+27) = LC3(row+start,col);
            l_stance_kinematics_data(row,col+30) = LFCC(row+start,col);
            l_stance_kinematics_data(row,col+33) = LFM2(row+start,col);
            l_stance_kinematics_data(row,col+36) = LTF(row+start,col);
            l_stance_kinematics_data(row,col+39) = LFMT(row+start,col);
            l_stance_kinematics_data(row,col+42) = RFLE(row+start,col);
            l_stance_kinematics_data(row,col+45) = RFME(row+start,col);
            l_stance_kinematics_data(row,col+48) = RT1(row+start,col);
            l_stance_kinematics_data(row,col+51) = RT2(row+start,col);
            l_stance_kinematics_data(row,col+54) = RT3(row+start,col);
            l_stance_kinematics_data(row,col+57) = RFAM(row+start,col);
            l_stance_kinematics_data(row,col+60) = RTAM(row+start,col);
            l_stance_kinematics_data(row,col+63) = RC1(row+start,col);
            l_stance_kinematics_data(row,col+66) = RC2(row+start,col);
            l_stance_kinematics_data(row,col+69) = RC3(row+start,col);
            l_stance_kinematics_data(row,col+72) = RFCC(row+start,col);
            l_stance_kinematics_data(row,col+75) = RFM2(row+start,col);
            l_stance_kinematics_data(row,col+78) = RTF(row+start,col);
            l_stance_kinematics_data(row,col+81) = RFMT(row+start,col);
            l_stance_kinematics_data(row,col+84) = RASIS(row+start,col);
            l_stance_kinematics_data(row,col+87) = LASIS(row+start,col);
            l_stance_kinematics_data(row,col+90) = RPSIS(row+start,col);
            l_stance_kinematics_data(row,col+93) = LPSIS(row+start,col);
        end
    end
catch %unlabeled
    l_stance_kinematics_data = zeros(finish-start,96);
    l_f_k_data = zeros(finish-start,97);
    start_frame1 = 1;
    start_frame2 = 1;
    end_frame1 = 1;
    end_frame2 = 1;
    return
end


% Save unmirrored left stance data
if plate1 == 'L'
    unmmirrored_left_data = zeros(end_frame1-start_frame1+20, 97);
    for row = 1:end_frame1-start_frame1+20
        unmmirrored_left_data(row,97) = binary_force1(row+start_frame1-11,1);
        for col = 1:96
            unmmirrored_left_data(row,col) = l_stance_kinematics_data(row+start_frame1-11,col);
        end
    end
end

if plate2 == 'L'
    unmmirrored_left_data = zeros(end_frame2-start_frame2+20, 97);
    for row = 1:end_frame2-start_frame2+20
        unmmirrored_left_data(row,97) = binary_force2(row+start_frame2-11,1);
        for col = 1:96
            unmmirrored_left_data(row,col) = l_stance_kinematics_data(row+start_frame2-11,col);
        end
    end
end


% For left-sided stance mirror the inputs

[mirrored_marker_data, force] = Mirror_input_data(l_stance_kinematics_data, l_force);

l_mirrored = zeros(finish-start,96);
for row = 1:finish-start
    for col = 1:96
        l_mirrored(row,col) = mirrored_marker_data(row,col);
    end
end

% This scatter graph can be used to check coordinates have been correctly
% mirrored. Change 'row' for different frame.

%row = 11;
%for coord = 1:32
%    x(coord) = l_mirrored(row,coord*3-2);
%    y(coord) = l_mirrored(row,coord*3-1);
%%end
%scatter3(x,y,z)

% Now combine the kinematic data with force data for the corresponding
% pressure plates, assign the matrices L/R, and shorten them to 10 0s
% either side of force plate readings

if plate1 == 'L'
    l_f_k_data = zeros(end_frame1-start_frame1+20, 97);
    for row = 1:end_frame1-start_frame1+20
        l_f_k_data(row,97) = binary_force1(row+start_frame1-11,1);
        for col = 1:96
            l_f_k_data(row,col) = l_mirrored(row+start_frame1-11,col);
        end
    end
end


if plate2 == 'L'
    l_f_k_data = zeros(end_frame2-start_frame2+20, 97);
    for row = 1:end_frame2-start_frame2+20
        l_f_k_data(row,97) = binary_force2(row+start_frame2-11,1);
        for col = 1:96
            l_f_k_data(row,col) = l_mirrored(row+start_frame2-11,col);
        end
    end
end


