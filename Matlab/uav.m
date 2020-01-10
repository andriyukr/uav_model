function mae = uav_ann_noor(params)
% UAV_ANN_NOOR Model of unmanned aerial vehicle with fuzzy neural network controller.
%   UAV_ANN_NOOR(PARAMS) returns a mean absolute error (MAE) for given
%   parameters in PARAMS (smaller is MAE, better is the performance).
%   PARAMS is a 30-element vector that determines the parameters of
%   gaussian membership functions. The range for the values in PARAMS(1:9)
%   is [-1, 1]. The range for the values in PARAMS(10:15) is [0, 1]. The
%   range for the values in PARAMS(16:24) is [-1, 1]. The range for the
%   values in PARAMS(25:30) is [0, 1].
% 
%   For example:
% 
%       params = [(rand(1, 9) - 0.5) / 100, 0.01, 0.01, 0.01, 0, 0, 0, (rand(1, 9) - 0.5) / 100, 0.01, 0.01, 0.01, 0, 0, 0];
%       error = uav_ann_noor(params)
% 
%   Copyright 2018 Andriy.

%% Initialize state

kend = 100000;
x(1:kend,1:12) = 0;
t = 0.001;

%% Initialize gains

Kpx = 0.1;
Kdx = 0.1;
Kpy = 0.1;
Kdy = 0.1;
Kpz = 40;
Kdz = 12;
Kpp = 30;
Kdp = 5;
Kpt = 30;
Kdt = 5;
Kpps = 30;
Kdps = 5;

%% Initialize constants

Ixx = 8.1*10^(-3);  % Quadrotor moment of inertia around X axis
Iyy = 8.1*10^(-3);  % Quadrotor moment of inertia around Y axis
Izz = 14.2*10^(-3);  % Quadrotor moment of inertia around Z axis
Jtp = 104*10^(-6);  % Total rotational moment of inertia around the propeller axis
m = 1;  % Mass of the Quadrotor in Kg
g = 9.81;   % Gravitational acceleration

%% Initial values

Xinit = 0;
Yinit = 0;
Zinit = 0;
Phiinit = 0;pi/6;
Thetainit = 0;pi/4;
Psiinit = 0;

%% Desired values

Xd = 2*cos(t*(1:kend));1*ones(1, kend);
Yd = 2*sin(t*(1:kend));-5*ones(1, kend);
Zd = 10 * ones(1, kend);
Psid = 0 * ones(1, kend);

%% High-Level ANN parameters

ANN_HIGH = 1;

w_high = [params(1), params(2), params(3); params(4), params(5), params(6); params(7), params(8), params(9)];
alpha_high = [params(10); params(11); params(12)];
gamma_high = [params(13); params(14); params(15)];

%% Low-Level ANN parameters

ANN_LOW = 1;

w_low = [params(16), params(17), params(18); params(19), params(20), params(21); params(22), params(23), params(24)];
alpha_low = [params(25); params(26); params(27)];
gamma_low = [params(28); params(29); params(30)];

%% Control loop

x(1,1) = Xinit;
x(1,2) = 0;
x(1,3) = Yinit;
x(1,4) = 0;
x(1,5) = Zinit;
x(1,6) = 0;
x(1,7) = Phiinit;
x(1,8) = 0;
x(1,9) = Thetainit;
x(1,10) = 0;
x(1,11) = Psiinit;
x(1,12) = 0;

% Y_high = [0; 0; 0];
% Y_low = [0; 0; 0];
% Attituded = [0 0 0];
% Alpha_high = [alpha_high'];
% W_high = [reshape(w_high, [1,9])];
% Alpha_low = [alpha_low'];

for k=1:kend-1
    
    %% Compute position errors
    
    ex = cos(x(k,11)) * (Xd(k) - x(k,1)) - sin(x(k,11)) * (Yd(k) - x(k,3));
    edx = cos(x(k,11)) * (- x(k,2)) - sin(x(k,11)) * (- x(k,4));
    
    ey = -sin(x(k,11)) * (Xd(k) - x(k,1)) - cos(x(k,11)) * (Yd(k) - x(k,3));
    edy = -sin(x(k,11)) * (- x(k,2)) - cos(x(k,11)) * (- x(k,4));
    
    ez = Zd(k) - x(k,5);
    edz = - x(k,6);
    
    error_high = [ex, edx; ey, edy; ez, edz];
%     error_high = max(min(error_high, 1), -1);
    
    %% Position controller
    
    Thetad = Kpx * ex + Kdx * edx; %
    Phid = Kpy * ey + Kdy * edy; %
    Thrust = m*(g + Kpz * ez + Kdz * edz)/(cos(x(k,9))*cos(x(k,7)));   % Total Thrust on the body along z-axis
      
    %% High-Level: Compute the output
     
    if(ANN_HIGH ~= 0)
        error_high = error_high ./ sum(abs(error_high), 2);
        y = sum(w_high .* [error_high, [-1; -1; -1]], 2);
    else
        y = [0; 0; 0];
    end
    
%     Y_high = [Y_high, y];
    
    %% High-Level: Update the parameters
      
    delta = [Thetad; Phid; Thrust - m*g] - y; %%%%%%%%%%%  calculate delta for the output layer  %%%%
    delta_w = alpha_high .* delta .* [error_high, [-1; -1; -1]]; %  calculate the delta weight matrix for the output layer %
    w_high = w_high + delta_w; %%%%%%%%  uptade weight matrix for the output layer  %%
    alpha_high = alpha_high + 2 * gamma_high .* abs([Thetad; Phid; Thrust - m*g] - y);
          
    %     W_high = [W_high; reshape(w_high, [1,9])];
    
    %% Attitude references
    
    Thetad = Thetad + y(1);
    Phid = Phid + y(2);
    Thrust = Thrust + y(3);
    
    Thetad = max([min([Thetad pi/2]) -pi/2]); %
    Phid = max([min([Phid pi/2]) -pi/2]); %
    
%     Attituded = [Attituded; Thetad Phid Thrust];
    
    %% Compute attitude errors
    
    ephi = Phid - x(k,7);
    edphi = - x(k,8);
    
    etheta = Thetad - x(k,9);
    edtheta = - x(k,10);
    
    epsi = Psid(k) - x(k,11);
    edpsi = - x(k,12);
    
    error_low = [ephi edphi; etheta edtheta; epsi edpsi];
%     error_low(:, 1) = max(min(error_low(:, 1), pi/2), -pi/2);
%     error_low(:, 2) = max(min(error_low(:, 2), 1), -1);
    
    %% Attitude controller
    
    tau_phi = Kpp * ephi + Kdp * edphi; % Roll input
    tau_theta = Kpt * etheta + Kdt * edtheta; % Pitch input
    tau_psi = Kpps * epsi + Kdps * edpsi; % Yawing moment
       
    %% Low-Level: Compute the output

    if(ANN_LOW ~= 0)
        error_low = error_low ./ (sum(abs(error_low), 2) + 0.001);
        y = sum(w_low .* [error_low, [-1; -1; -1]], 2);
    else
        y = [0; 0; 0];
    end
    
%     Y_low = [Y_low, y];
    
    %% Low-Level: Update the parameters
      
    delta = [tau_phi; tau_theta; tau_psi] - y; %%%%%%%%%%%  calculate delta for the output layer  %%%%
    delta_w = alpha_low .* delta .* [error_low, [-1; -1; -1]]; %  calculate the delta weight matrix for the output layer %
    w_low = w_low + delta_w; %%%%%%%%  uptade weight matrix for the output layer  %%
    alpha_low = alpha_low + 2 * gamma_low .* abs([tau_phi; tau_theta; tau_psi]);
       
    %     W_low = [W_low; reshape(w_low, [1,9])];
    
    %% Control inputs
    
    tau_phi = tau_phi + y(1);
    tau_theta = tau_theta + y(2);
    tau_psi = tau_psi + y(3);
    
    U = [Thrust tau_phi tau_theta tau_psi];
    
    %% System dynamics
    
    xdot(1) = x(k,2); % Xdot
    xdot(2) = (sin(x(k,11))*sin(x(k,7)) + cos(x(k,11))*sin(x(k,9))*cos(x(k,7)))*(U(1)/m);    % Xdotdot
    xdot(3) = x(k,4); % Ydot
    xdot(4) = (-cos(x(k,11))*sin(x(k,7)) + sin(x(k,11))*sin(x(k,9))*cos(x(k,7)))*(U(1)/m);	% Ydotdot
    xdot(5) = x(k,6); % Zdot
    xdot(6) = - g + (cos(x(k,9))*cos(x(k,7)))*(U(1)/m);    % Zdotdot
    xdot(7) = x(k,8); % phydot
    xdot(8) = ((Iyy - Izz)/Ixx)*x(k,10)*x(k,12) - (Jtp/Ixx)*x(k,10) + (U(2)/Ixx); % pdot = phydotdot
    xdot(9) = x(k,10);    % thetadot
    xdot(10) = ((Izz - Ixx)/Iyy)*x(k,8)*x(k,12) + (Jtp/Iyy)*x(k,8) + (U(3)/Iyy);	% qdot = thetadotdot
    xdot(11) = x(k,12);   % thetadot
    xdot(12) = ((Ixx - Iyy)/Izz)*x(k,8)*x(k,10) + (U(4)/Izz);	% rdot = psidotdot
    
    x(k + 1,:) = x(k,:) + t * xdot;
end

%% Errors

error = sqrt((Xd - x(:,1)').^2 + (Yd - x(:,3)').^2 + (Zd - x(:,5)').^2);
mae = mean(error);
% mse = mean(error.^2);
% rmse = sqrt(mean(error.^2));