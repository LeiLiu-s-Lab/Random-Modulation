% ----------------------------------------------------------------
% Test file: plot a figure "MSE v.s. number of iterations"
% Authors: Lei Liu, Shunqi Huang, Yuhao Chi, Yao Ge
% ----------------------------------------------------------------
% If you have any questions, please contact:
% lei_liu@zju.edu.cn
% ----------------------------------------------------------------
% If you use this code, please cite:
% [1] L. Liu, Y. Chi, S. Huang, "Random Modulation: Achieving Asymptotic 
% Replica Optimality over Arbitrary Norm-Bounded and Spectrally Convergent 
% Channel Matrices," 2025 IEEE International Symposium on Information Theory (ISIT).
% [2] Y. Chi, L. Liu, Y. Ge, X. Chen, Y. Li, and Z. Zhang, "Interleave Frequency 
% Division Multiplexing," IEEE Wireless Communications Letters, 
% vol. 13, no. 7, pp. 1963 - 1967, 2024.
% [3] L. Liu, S. Huang, and B. M. Kurkoski, "Memory AMP," 
% IEEE Transactions on Information Theory, vol. 68, no. 12, pp. 8015-8039, 2022.

%% Parameters
clc; clear;
rng('default');
% Multipath channel parameters
P = 5;                      % Number of Path
delta_f = 1.5e4;            % Subcarrier spacing 
M = 32;                     % total delay span > channel's maximum delay
N = 32;                     % total Doppler span > channel's maximum Doppler shift
MN = M * N;                                       
vel = 100;                              % Velocity
dop = vel * (1e3/3600) * (4e9/3e8);     % Doppler frequency shift
index_D = 1;                            % 1 means Doppler shift
fs_N = 1;                               % >1 means oversampling
fs = fs_N * M * delta_f;                % Sampling rate
beta = 0.4;                             % Roll-off factor of raised-cosine filter
N_s = 2;                    % Number of transmit antennas
N_r = 2;                    % Number of received antennas
rho = 0;                    % MIMO correlation factor
N_y = N_r * MN;          
N_x = N_s * MN;     
SNR_dB = 13;
v_n = 1 / (10^(0.1*SNR_dB));
Num_sim = 100;
L = 3;                                  % damping length, used in MAMP
info = struct('type', "QPSK", 'mean', 0, 'var', 1);     % See 'Demodulator.m'
% Modulations
% (*) Here, for OFDM and AFDM, we use the joint modulation scheme. The data
% for all antennas is modulated collectively. 
ofdm_info = struct('type', "OFDM_j", 'N_x', N_x);       
otfs_info = struct('type', "OTFS", 'M', M, 'N', N, 'N_s', N_s);
Epsilon = N;                            % 0 (>0): integer (fractional) Doppler shift
Doppler_taps_max = round(dop*N/delta_f);
c1 = (2*(Doppler_taps_max+Epsilon)+1) / (2*N_x);
c2 = 1e-5;                              % c2 should be much smaller than 1/(2*N_x) 
afdm_info = struct('type', "AFDM_j", 'N_x', N_x, 'c1', c1, 'c2', c2);
% % Per-antenna OFDM and AFDM
% ofdm_info = struct('type', 'OFDM_p', 'MN', MN, 'N_s', N_s);
% c1 = (2*(Doppler_taps_max+Epsilon)+1) / (2*MN);
% afdm_info = struct('type', "AFDM_p", 'MN', MN, 'N_s', N_s, 'c1', c1, 'c2', c2);
iter_O = 10;
iter_M = 20;
MSE_ofdm = zeros(1, iter_O);
MSE_otfs = zeros(1, iter_O);
MSE_afdm = zeros(1, iter_O);
MSE_rm = zeros(1, iter_M);

%% Simulations
snr = SNR_dB;
parfor jj = 1 : Num_sim
    disp(jj);
    % QPSK signal
    s_da = binornd(1, 0.5, 2*N_x, 1);
    s = Bits_to_QPSK(s_da);
    % Time-domain multipath channel 
    H = Get_channel(M, N, N_r, N_s, rho, delta_f, fs, fs_N, P, index_D, dop, beta);
    [~, dia, V] = svd(H);               % necessary for OAMP
    dia = diag(dia);
    temp = sum(dia.^2) / N_x;
    H = H / sqrt(temp);                 % channel normalization 
    dia = dia / sqrt(temp);             % channel normalization 
    H = sparse(H);                      % sparse H
    % Gaussian noise
    n_re = normrnd(0, sqrt(v_n/2), [N_y, 1]); 
    n_im = normrnd(0, sqrt(v_n/2), [N_y, 1]);
    n = n_re + n_im * 1i;
    % OFDM, OTFS, AFDM
    x_ofdm = Modulations(s, ofdm_info, 0);
    y_ofdm = H * x_ofdm + n;
    x_otfs = Modulations(s, otfs_info, 0);
    y_otfs = H * x_otfs + n;
    x_afdm = Modulations(s, afdm_info, 0);
    y_afdm = H * x_afdm + n;
    % RM: y = H * Xi * s + n, Xi = Pi * F 
    % Pi is a random permutation, F is a fast transform
    index = randperm(N_x);
    rm_info = struct('type', "RM", 'rm_type', "fft", 'N_x', N_x, 'index', index);
    x_rm = Modulations(s, rm_info, 0);
    y_rm = H * x_rm + n;
    % CD-OAMP detector
    % No demodulation is performed at the receiver before CD-OAMP detector.
    % This is because the performance of CD-OAMP is invariant.
    [MSE, ~, ~] = CD_OAMP(H, V, s, y_ofdm, dia, v_n, iter_O, info, ofdm_info);
    MSE_ofdm = MSE_ofdm + MSE;
    [MSE, ~, ~] = CD_OAMP(H, V, s, y_otfs, dia, v_n, iter_O, info, otfs_info);
    MSE_otfs = MSE_otfs + MSE;
    [MSE, ~, ~] = CD_OAMP(H, V, s, y_afdm, dia, v_n, iter_O, info, afdm_info);
    MSE_afdm = MSE_afdm + MSE;
    % CD-MAMP detector
    [MSE, ~, ~] = CD_MAMP_e(H, s, y_rm, v_n, L, iter_M, info, rm_info);
    MSE_rm = MSE_rm + MSE;
end
MSE_ofdm = MSE_ofdm / Num_sim;
MSE_otfs = MSE_otfs / Num_sim;
MSE_afdm = MSE_afdm / Num_sim;
MSE_rm = MSE_rm / Num_sim;

%% plot figures
semilogy(0:iter_M, [1 MSE_rm], 'r-', 'LineWidth', 1.5);
hold on;
semilogy(0:iter_O, [1 MSE_ofdm], '-', 'LineWidth', 1.5);
semilogy(0:iter_O, [1 MSE_otfs], '-', 'LineWidth', 1.5);
semilogy(0:iter_O, [1 MSE_afdm], '-', 'LineWidth', 1.5);
legend('RM', 'ofdm', 'otfs', 'afdm');
xlabel('Number of iterations', 'FontSize', 11);
ylabel('MSE', 'FontSize', 11);
