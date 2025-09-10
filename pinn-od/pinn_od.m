%% 0) Data (times, sensor, RA/Dec). Replace with real observations if you have them.
mu = 398600.4418;                  % km^3/s^2 (Earth)
rng(1);

% Training times (48 hr, 30 samples randomly, like Fig. 2)
t0 = 0; tf = 48*3600;              % seconds
K  = 30;
tObs = sort(t0 + rand(K,1)*(tf - t0));

% Sensor in ECI (simple fixed site rotated to ECI per epoch is better; keep simple here)
sensorECI = @(t) [42164;0;0]*0 + [ -6378; 0; 0]; % placeholder; supply true ECI sensor pos per t
arcsec2rad = pi / (180 * 3600);   % 1 arcsecond in radians

% --- If you have truth ephemeris + thrust, you can generate synthetic RA/Dec like the paper.
% For now, we assume you already have measured RA/Dec (rad) in raObs/decObs and their sigmas.
[raObs, decObs, sigmaRA, sigmaDec] = deal(zeros(K,1), zeros(K,1), 0.5*arcsec2rad, 0.5*arcsec2rad); % fill or load

%% 1) Physics model: two-body (swap later with higher fidelity)
function a = a_phys(~, x, mu)
    r = x(1:3); v = x(4:6); %#ok<NASGU>
    a = -mu * r / (norm(r)^3);
end

%% 2) Thrust MLP g_theta(t, x) -> a_thrust (km/s^2), small magnitude
% Build a small MLP with two hidden layers (≈100 units, tanh) as in the paper (Sec. 3, "DNN Architecture").
layers = [
    featureInputLayer(1+6,"Name","in")                            % concat time + state
    fullyConnectedLayer(100,"Name","fc1")
    tanhLayer("Name","t1")
    fullyConnectedLayer(100,"Name","fc2")
    tanhLayer("Name","t2")
    fullyConnectedLayer(3,"Name","out")                           % ax, ay, az
];
net = dlnetwork(layers);

%% 3) Learnable initial conditions (or use alternating LS every N iters)
x0_guess = [42164; 0; 0; 0; 3.0746; 0];  % rough GEO seed (km, km/s). Replace with your estimator’s seed.
x0 = dlarray(x0_guess, "CB");            % 6x1 as column "batch" dim

%% 4) Differentiable RK4 integrator over arbitrary times (dlarray-safe)
function X = propagate_rk4(net, x0, tObs, mu)
    % Returns 6xK states at observation times. Steps with variable dt between samples.
    X = dlarray(zeros(6, numel(tObs)), "CB");
    x = x0;
    tprev = tObs(1);
    % ensure first sample: if tObs(1) ~= 0, we propagate from t=0 to tObs(1)
    tgrid = [0; tObs(:)];
    x = x0;
    t = tgrid(1);
    for i = 2:numel(tgrid)
        dt = tgrid(i) - t;
        % substep RK4 with Nsub (stability for long dt). Choose e.g., 20 steps per 1hr.
        Nsub = max(1, ceil(abs(dt)/(180))); % 3-min steps
        h = dt / Nsub;
        for s = 1:Nsub
            k1 = f_total(net, t,   x, mu);
            k2 = f_total(net, t+h/2, x + h/2*k1, mu);
            k3 = f_total(net, t+h/2, x + h/2*k2, mu);
            k4 = f_total(net, t+h,   x + h*k3,   mu);
            x  = x + (h/6)*(k1 + 2*k2 + 2*k3 + k4);
            t  = t + h;
        end
        if i>1
            X(:,i-1) = x;
        end
    end
end

function xdot = f_total(net, t, x, mu)
    % xdot = [v; a_phys + a_thrust]
    v = x(4:6);
    a = a_phys(t, x, mu) + a_thrust(net, t, x);
    xdot = [v; a];
end

function ath = a_thrust(net, t, x)
    % Normalize inputs for stability (time in hours, state scaled).
    tin = single(t/3600);                     % hours
    xs  = stateScale(x);                      % simple scaling helper below
    ins = dlarray([tin; xs], "CB");
    ath = forward(net, ins);                  % km/s^2
    % Optional tiny magnitude clamp or tanh scaling if needed
end

function xs = stateScale(x)
    % rough scales (GEO): r~4e4 km, v~3 km/s
    s = [5e4;5e4;5e4; 10;10;10];
    xs = x ./ s;
end

%% 5) Measurement model: RA/Dec from ECI relative geometry
function [ra, dec] = predictAngles(X, tObs, sensorECI)
    % X: 6xK, positions in columns
    K = size(X,2);
    ra  = dlarray(zeros(1,K), "CB");
    dec = dlarray(zeros(1,K), "CB");
    for k=1:K
        r_sc = X(1:3,k);
        r_s  = sensorECI(tObs(k));
        rho  = r_sc - r_s;                       % line-of-sight in ECI
        x = rho(1); y = rho(2); z = rho(3);
        ra(1,k)  = atan2(y,x);
        dec(1,k) = atan2(z, sqrt(x^2 + y^2));
    end
    % wrap RA to [0, 2π) if you like — for loss, difference via sin/cos is safer
end

function e = angResidual(pred, meas)
    % robust “circular” residual using sin/cos to avoid RA wrap issues
    e = atan2(sin(pred - meas), cos(pred - meas));
end

arcsec2rad = pi/(180*3600);

%% 6) Loss: weighted MSE on RA/Dec (like paper’s RA/Dec MSE; Sec. 3)
function [L, grads, aux] = modelLoss(net, x0, tObs, sensorECI, raObs, decObs, sigRA, sigDec, mu)
    % forward propagate
    X = propagate_rk4(net, x0, tObs, mu);
    [raHat, decHat] = predictAngles(X, tObs, sensorECI);

    % residuals (angle-safe)
    eRA  = angResidual(raHat,  raObs);
    eDec = angResidual(decHat, decObs);

    % WLS (use provided sigmas, ~0.5 arcsec in the paper’s sim; Fig. 6/7, Table 2)
    L = mean( (eRA./sigRA').^2 + (eDec./sigDec').^2 );

    % backprop
    learnables = [net.Learnables.Value; x0];
    gradients = dlgradient(L, net.Learnables.Value, x0);
    grads.net = gradients(1:end-1);
    grads.x0  = gradients(end);
    aux.X = X;
end

%% 7) Training loop (Adam + LR decay; Table 1 uses 20k iters, lr=0.003, step 100, factor 0.98)
numEpochs = 10000;        % start smaller, scale up
lr = 3e-3;                % initial
decayStep = 100; decayFactor = 0.98;

trailingAvg = []; trailingSq = [];
trailingAvgX0 = []; trailingSqX0 = [];

for epoch = 1:numEpochs
    [L, grads, ~] = dlfeval(@modelLoss, net, x0, tObs, sensorECI, dlarray(raObs), dlarray(decObs), dlarray(sigmaRA*ones(K,1)), dlarray(sigmaDec*ones(K,1)), mu);

    % --- Adam updates: network
    [net, trailingAvg, trailingSq] = adamupdate(net, grads.net, trailingAvg, trailingSq, epoch, lr, 0.9, 0.999);
    % --- Adam update: initial conditions (optional — or hold IC and do LS every N)
    [x0, trailingAvgX0, trailingSqX0] = adamupdate(x0, grads.x0, trailingAvgX0, trailingSqX0, epoch, lr, 0.9, 0.999);

    % LR decay
    if mod(epoch,decayStep)==0
        lr = lr*decayFactor;
    end

    if mod(epoch,100)==0
        fprintf('Epoch %d  Loss %.3f\n', epoch, gather(extractdata(L)));
    end

    % --- Optional: alternating least squares on ICs every N (paper did every 100) ---
    % if mod(epoch,100)==0
    %     x0 = batchLeastSquaresIC(net, x0, tObs, sensorECI, raObs, decObs, mu);
    % end
end

%% 8) After training: propagate beyond fit span for extrapolation (Table 2; Fig. 8)
% Build a dense time grid for 20 days and compare to truth ephemeris if you have it.
function x0_new = batchLeastSquaresIC(net, x0, tObs, sensorECI, raObs, decObs, mu)
    x0v = gather(extractdata(x0));
    fun = @(x0try) double(gLoss(x0try));
    options = optimoptions('lsqnonlin','Display','off');
    x0_new = dlarray(lsqnonlin(fun, x0v, [], [], options), "CB");

    function r = gLoss(x0try)
        X = propagate_rk4(net, dlarray(x0try,"CB"), tObs, mu);
        [raHat, decHat] = predictAngles(X, tObs, sensorECI);
        eRA  = angResidual(raHat,  raObs);
        eDec = angResidual(decHat, decObs);
        r = [gather(extractdata(eRA))'; gather(extractdata(eDec))']';
    end
end
