%% Clear

clc
clear all
close all

%% Parameters

kend = 20000;
dt = 0.001;

%% Initialize gains

Kp_x = 1;
Ki_x = 0.1;
Kd_x = 2;

Kp_y = 1;
Ki_y = 0.1;
Kd_y = 2;

Kp_z = 50;
Ki_z = 0.1;
Kd_z = 10;

Kp_roll = 30;
Ki_roll = 0.1;
Kd_roll = 5;

Kp_pitch = 30;
Ki_pitch = 0.1;
Kd_pitch = 5;

Kp_yaw = 30;
Ki_yaw = 0.1;
Kd_yaw = 5;

%% Initialize constants

Ixx = 8.1*10^(-3);  % Quadrotor moment of inertia around X axis
Iyy = 8.1*10^(-3);  % Quadrotor moment of inertia around Y axis
Izz = 14.2*10^(-3);  % Quadrotor moment of inertia around Z axis
Jtp = 104*10^(-6);  % Total rotational moment of inertia around the propeller axis
b = 54.2*10^(-5);  % Thrust factor
d = 1.1*10^(-6);  % Drag factor
l = 0.2;  % Distance to the center of the Quadrotor
m = 1;  % Mass of the Quadrotor in Kg
g = 9.81;   % Gravitational acceleration

%% Initial state

x_init = 0;
y_init = 0;
z_init = 0;
roll_init = pi/6;
pitch_init = pi/4;
yaw_init = pi/2;

%% Reference values

t = dt*(1:kend);

x_ref = 2*cos(t);% 2*t;
dx_ref = -2*sin(t);% 0*t;

y_ref = 2*sin(t);% -5*t;
dy_ref = 2*cos(t);% 0*t;

z_ref = 0.1*t;
dz_ref = 0.1*ones(kend,1);

yaw_ref = 0*t;
dyaw_ref = 0*ones(kend,1);

Attituded = zeros(kend,3);

%% Control loop

state = zeros(kend, 12);

state(1,1) = x_init;
state(1,2) = y_init;
state(1,3) = z_init;
state(1,4) = 0;
state(1,5) = 0;
state(1,6) = 0;
state(1,7) = roll_init;
state(1,8) = pitch_init;
state(1,9) = yaw_init;
state(1,10) = 0;
state(1,11) = 0;
state(1,12) = 0;

input = zeros(kend, 4);

for k = 1:kend - 1
    
    %% Compute position errors
    
    e_x = cos(state(k,9)) * (x_ref(k) - state(k,1)) - sin(state(k,9)) * (y_ref(k) - state(k,2));
    e_dx = cos(state(k,9)) * (dx_ref(k) - state(k,4)) - sin(state(k,9)) * (dy_ref(k) - state(k,5));
    
    e_y = -sin(state(k,9)) * (x_ref(k) - state(k,1)) - cos(state(k,9)) * (y_ref(k) - state(k,2));
    e_dy = -sin(state(k,9)) * (dx_ref(k) - state(k,4)) - cos(state(k,9)) * (dy_ref(k) - state(k,5));
    
    e_z = z_ref(k) - state(k,3);
    e_dz = dz_ref(k) - state(k,6);
    
    error_high = [e_x, e_dx; e_y, e_dy; e_z, e_dz];
    
    %% Position controller
    
    roll_ref = Kp_x * e_x + Kd_x * e_dx; %
    pitch_ref = Kp_y * e_y + Kd_y * e_dy; %
    thrust = m*(g + Kp_z * e_z + Kd_z * e_dz)/(cos(state(k,7))*cos(state(k,8)));   % Total Thrust on the body along z-axis
    
    roll_ref = max([min([roll_ref pi/2]) -pi/2]); %
    pitch_ref = max([min([pitch_ref pi/2]) -pi/2]); %
    
    Attituded(k,:) = [roll_ref pitch_ref thrust];
    
    %% Compute attitude errors
    
    e_roll = pitch_ref - state(k,7);
    e_droll = - state(k,10);
    
    e_pitch = roll_ref - state(k,8);
    e_dpitch = - state(k,11);
    
    e_yaw = yaw_ref(k) - state(k,9);
    e_dyaw = - state(k,12);
    
    error_low = [e_roll e_droll; e_pitch e_dpitch; e_yaw e_dyaw];
    
    %% Attitude controller
    
    tau_roll = Kp_roll * e_roll + Kd_roll * e_droll; % Roll rate
    tau_pitch = Kp_pitch * e_pitch + Kd_pitch * e_dpitch; % Pitch rate
    tau_yaw = Kp_yaw * e_yaw + Kd_yaw * e_dyaw; % Yaw rate
    
    input(k,:) = [thrust tau_roll tau_pitch tau_yaw];
    
    %% System dynamics
    
    dstate(1) = state(k,4);                                                                                             % x_dot
    dstate(2) = state(k,5);                                                                                             % y_dot
    dstate(3) = state(k,6);                                                                                             % z_dot
    dstate(4) = (sin(state(k,9))*sin(state(k,7)) + cos(state(k,9))*sin(state(k,8))*cos(state(k,7)))*(input(k,1)/m);     % x_dotdot
    dstate(5) = (-cos(state(k,9))*sin(state(k,7)) + sin(state(k,9))*sin(state(k,8))*cos(state(k,7)))*(input(k,1)/m);    % y_dotdot
    dstate(6) = (cos(state(k,8))*cos(state(k,7)))*(input(k,1)/m) - g;                                                   % z_dotdot
    dstate(7) = state(k,10);                                                                                            % roll_dot
    dstate(8) = state(k,11);                                                                                            % pitch_dot
    dstate(9) = state(k,12);                                                                                            % yaw_dot
    dstate(10) = ((Iyy - Izz)/Ixx)*state(k,11)*state(k,12) - (Jtp/Ixx)*state(k,11) + (input(k,2)/Ixx);                  % roll_dotdot
    dstate(11) = ((Izz - Ixx)/Iyy)*state(k,10)*state(k,12) + (Jtp/Iyy)*state(k,10) + (input(k,3)/Iyy);                  % pitch_dotdot
    dstate(12) = ((Ixx - Iyy)/Izz)*state(k,10)*state(k,11) + (input(k,4)/Izz);                                          % yaw_dotdot
    
    state(k + 1,:) = state(k,:) + dt * dstate;
    
    if rem(k,1000) == 0
        disp([num2str(k/(kend - 1)*100), '%']);
    end
end
disp('100%');

%% Plots

figure('Name', 'x tracking', 'NumberTitle', 'off');
h = plot(dt*(0:(kend-1)), x_ref, dt*(0:(kend-1)), state(:, 1));
set(h, 'LineWidth', 2);
legend('desired', 'actual');

figure('Name', 'y tracking', 'NumberTitle', 'off');
h = plot(dt*(0:(kend-1)), y_ref, dt*(0:(kend-1)), state(:, 2));
set(h, 'LineWidth', 2);
legend('desired', 'actual');

figure('Name', 'z tracking', 'NumberTitle', 'off');
h = plot(dt*(0:(kend-1)), z_ref, dt*(0:(kend-1)), state(:, 3));
set(h, 'LineWidth', 2);
legend('desired', 'actual');

figure('Name', '3D tracking', 'NumberTitle', 'off');
h = plot3(x_ref, y_ref, z_ref, state(:, 1), state(:, 2), state(:, 3));
set(h, 'LineWidth', 2);
legend('desired', 'actual');

% figure('Name', 'Control Signals', 'NumberTitle', 'off');
% h = plot([(Attituded - Y_high'), Y_high']);
% set(h, 'LineWidth', 2);
% legend('PID_x', 'PID_y', 'PID_z', 'ANN_x', 'ANN_y', 'ANN_z');

% figure('Name', 'roll tracking', 'NumberTitle', 'off');
% h = plot(dt*(0:(kend-1)), Attituded(:,1) / pi * 180, dt*(0:(kend-1)), state(:, 7) / pi * 180);
% set(h, 'LineWidth', 2);
% legend('desired', 'actual');
%
% figure('Name', 'pitch tracking', 'NumberTitle', 'off');
% h = plot(dt*(0:(kend-1)), Attituded(:,2) / pi * 180, dt*(0:(kend-1)), state(:, 8) / pi * 180);
% set(h, 'LineWidth', 2);
% legend('desired', 'actual');
%
% figure('Name', 'yaw tracking', 'NumberTitle', 'off');
% h = plot(dt*(0:(kend-1)), Psid / pi * 180, dt*(0:(kend-1)), state(:, 9) / pi * 180);
% set(h, 'LineWidth', 2);
% legend('desired', 'actual');

%% Performance

v_max = max(sqrt(state(:,4).^2 + state(:,5).^2 + state(:,6).^2))

error = sqrt((x_ref - state(:,1)').^2 + (y_ref - state(:,2)').^2 + (z_ref - state(:,3)').^2);
mae = mean(error)
rmse = sqrt(mean(error.^2))

%% Save data

state = [state(:,1:6) eul2quat(state(:,7:9)) state(:,10:12)]; % caonvert Euler angles to quaternions

save('dataUAV', 'state', 'input');