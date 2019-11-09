wd = pwd;


load_file = sprintf('WALKING 3.c3d');
btk_load = [load_file];

raw_markers_unmirrored = 'raw_markers_unmirrored.txt';

raw_markers_opposite = 'raw_markers_opposite.txt';


output_filename = 'marker_and_force_data.txt';

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

%frame at which events happen in kinematics matrix
right_foot_strike = (events.Right_Foot_Strike(1)*sample_freq) - start;
right_foot_off = (events.Right_Foot_Off(1)*sample_freq) - start;

%if earlier foot-off picked out, then frame will be negative
if right_foot_off <= right_foot_strike
    right_foot_off = (events.Right_Foot_Off(2)*sample_freq) - start;
        
end

%frame at which events happen in force plate matrix size
    r_strike = round(right_foot_strike*down_rate);
    r_off = round(right_foot_off*down_rate);
    
%frame at which events happen in kinematics matrix
left_foot_strike = (events.Left_Foot_Strike(1)*sample_freq) - start;
left_foot_off = (events.Left_Foot_Off(1)*sample_freq) - start;

%if earlier foot-off picked out, then frame will be negative
if left_foot_off <= left_foot_strike
    left_foot_off = (events.Left_Foot_Off(2)*sample_freq) - start;
        
end

%frame at which events happen in force plate matrix size
    l_strike = round(left_foot_strike*down_rate);
    l_off = round(left_foot_off*down_rate);
    

% Force plate information
COP1  = grw(1).P; 
ex_force1 = grw(1).F;
COP2  = grw(2).P; 
ex_force2 = grw(2).F;


%determine which force plate is right/left
if (ex_force1(r_strike-50,1) == 0)&&(ex_force1(r_strike+50,1) ~= 0)
    plate1 = 'R';
elseif (ex_force2(r_strike-50,1) == 0)&&(ex_force2(r_strike+50,1) ~= 0)
    plate2 = 'R';
end

if (ex_force1(l_strike-50,1) == 0)&&(ex_force1(l_strike+50,1) ~= 0)
    plate1 = 'L';
end
if (ex_force2(l_strike-50,1) == 0)&&(ex_force2(l_strike+50,1) ~= 0)
    plate2 = 'L';
end


%create matrices which display a 1 or 0, depending on whether a force is
%measured
binary_force1 = zeros((finish-start)*down_rate, 1);
binary_force2 = zeros((finish-start)*down_rate, 1);

for row = 1:(finish-start)*down_rate
        if ex_force1(row,1) ~= 0
           binary_force1(row,1) = 1;
        end
        if ex_force2(row,1) ~= 0
           binary_force2(row,1) = 1;
        end
end

r_binary_force = zeros(r_off-r_strike+40, 1);
l_binary_force = zeros(l_off-l_strike+40, 1);

if plate1 == 'R'
    for row = 1:r_off-r_strike+40
        r_binary_force(row,1) = binary_force1(row+r_strike,1);
    end
end

if plate2 == 'L'
    for row = 1:l_off-l_strike+40
        l_binary_force(row,1) = binary_force2(row+l_strike,1);
    end
end

if plate2 == 'R'
    for row = 1:r_off-r_strike+40
        r_binary_force(row,1) = binary_force2(row+r_strike,1);
    end
end

if plate1 == 'L'
    for row = 1:l_off-l_strike+40
        l_binary_force(row,1) = binary_force1(row+l_strike,1);
    end
end

