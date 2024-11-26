function [xm,Pm,xp,Pp,beta]=JPDA_wkf_light(z,xm,Pm,H,F,R,Q,Pd,flag)
% Joint probablistic data association wrapped kalman filter
% (1) Fortmann, Thomas, Yaakov Bar-Shalom, and Molly Scheffe. "Sonar tracking of multiple targets using joint probabilistic data association." IEEE journal of Oceanic Engineering 8.3 (1983): 173-184.
% (2) Traa, Johannes, and Paris Smaragdis. "A wrapped Kalman filter for azimuthal speaker tracking." IEEE Signal Processing Letters 20.12 (2013): 1257-1260.
%
% inputs 
%--------
% Q IS THE NUMBER OF SOURCES
% ---------------------------
% z - single observation, i.e., DOA estimate in the TF domain (1x1)
% xm - state prediction at time t (2XQ)
% Pm - covariance predication at time t (2X2XQ)
% H - measurement matrix (2X1)
% F - transition matrix (2X2)
% R - measurement error covariance matrix (1X1)
% Q - process noise covariance matrix (2X2)
% Pd - probability of detection (1XQ)
%
% outputs
% -------
% xm - state prediction at time t+1 (2XQ)
% Pm - covariance predication at time t+1 (2X2XQ)
% xp - state estimation at time t (2XQ)
% Pp - covariance estimation at time t (2X2XQ)
% beta - association probabilities (1XQ)

% Hanan Beit-On, June 2022


% calculate association probabilities 
for q=1:2
    S = H*Pm(:,:,q)*H.'+ R; %innovation covariance
    eta = normpdf(z + 2*pi*(-1:1),xm(1,q),sqrt(R)) ./ sum(normpdf(z + 2*pi*(-100:100),xm(1,q),sqrt(R))); % WKF innovation weights
    gt(q) = (z + 2*pi*(-1:1) - xm(1,q))*eta.'; % WKF innovation
    beta(q) = normpdf(gt(q),0,sqrt(S)) * Pd(q)^2; % posterior association probabilities [1] eq. (3.18)
end

beta = beta./sum(beta); %normalize beta to ensure beta(1) + beta(2) = 1
beta = double(beta);


for q=1:2
    % correct
    K = Pm(:,:,q)*H.'/(H*Pm(:,:,q)*H.'+ R);
    gt_weighted(q) = beta(q) * gt(q); %JPDA inovation is a weighted sum of the inovations
    xp(:,q) = xm(:,q) + K*gt_weighted(q);
    xp(1,q) = wrapToPi(xp(1,q));
    Pp(:,:,q) = Pm(:,:,q) - K*H*Pm(:,:,q);

    % predict
    xm(:,q) = F*xp(:,q);
    xm(1,q) = wrapToPi(xm(1,q));
    Pm(:,:,q) = F*Pp(:,:,q)*F.' + Q(:,:,q);
end

end
