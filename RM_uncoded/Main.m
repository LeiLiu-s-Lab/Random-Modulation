% ----------------------------------------------------------------
% Uncoded case: random modulation (RM) v.s. OFDM/OTFS/AFDM
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
rng('shuffle');
% Multipath channel parameters
P = 5;                      % Number of Path
delta_f = 1.5e4;            % Subcarrier spacing 
M = 32;                     % total delay span > channel's maximum delay
N = 64;                     % total Doppler span > channel's maximum Doppler shift
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

%% Simulations
SNR_dB = 4:2:14;
len_snr = length(SNR_dB);
BER_ofdm = zeros(1, len_snr);
BER_otfs = zeros(1, len_snr);
BER_afdm = zeros(1, len_snr);
BER_rm = zeros(1, len_snr);
Num_sim = [100, 100, 100, 100, 1000, 1000];
% Parallel computing
poolobj = gcp('nocreate');     
if isempty(poolobj)
    poolsize = 0;
    CoreNum = 12;
    parpool(CoreNum);
end
for ii = 1 : len_snr
    snr = SNR_dB(ii);
    v_n = 1 / (10^(0.1*snr));
    [E_1, E_2, E_3, E_4] = deal(0);
    fprintf('---------------SNR: %ddB--------------- \n', snr);
    parfor jj = 1 : Num_sim(ii)
        % QPSK signal
        d = binornd(1, 0.5, 2*N_x, 1);
        s = Bits_to_QPSK(d);
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
        [~, ~, s_ofdm] = CD_OAMP(H, V, s, y_ofdm, dia, v_n, iter_O, info, ofdm_info);
        [~, ~, s_otfs] = CD_OAMP(H, V, s, y_otfs, dia, v_n, iter_O, info, otfs_info);
        [~, ~, s_afdm] = CD_OAMP(H, V, s, y_afdm, dia, v_n, iter_O, info, afdm_info);
        % CD-MAMP detector
        [~, ~, s_rm] = CD_MAMP_e(H, s, y_rm, v_n, L, iter_M, info, rm_info);
        % Hard decision (bits)
        d_ofdm = QPSK_to_bits(s_ofdm);
        E_1 = E_1 + sum(d_ofdm~=d);
        d_otfs = QPSK_to_bits(s_otfs);
        E_2 = E_2 + sum(d_otfs~=d);
        d_afdm = QPSK_to_bits(s_afdm);
        E_3 = E_3 + sum(d_afdm~=d);
        d_rm = QPSK_to_bits(s_rm);
        E_4 = E_4 + sum(d_rm~=d);
    end
    BER_ofdm(ii) = E_1 / Num_sim(ii) / (2*N_x);
    BER_otfs(ii) = E_2 / Num_sim(ii) / (2*N_x);
    BER_afdm(ii) = E_3 / Num_sim(ii) / (2*N_x);
    BER_rm(ii) = E_4 / Num_sim(ii) / (2*N_x);
    fprintf('BER: \n');
    fprintf('OFDM + OAMP: %.6f \n', BER_ofdm(ii))
    fprintf('OTFS + OAMP: %.6f \n', BER_otfs(ii))
    fprintf('AFDM + OAMP: %.6f \n', BER_afdm(ii))
    fprintf('RM + MAMP %.6f \n', BER_rm(ii))
    fprintf('------------------------------ \n')
end

%% plot figures
semilogy(SNR_dB, BER_rm, 'r-', 'LineWidth', 1.5);
hold on;
semilogy(SNR_dB, BER_ofdm, '-', 'LineWidth', 1.5);
semilogy(SNR_dB, BER_otfs, '-', 'LineWidth', 1.5);
semilogy(SNR_dB, BER_afdm, '-', 'LineWidth', 1.5);
legend('RM', 'ofdm', 'otfs', 'afdm');
xlabel('SNR (dB)', 'FontSize', 11);
ylabel('BER', 'FontSize', 11);
