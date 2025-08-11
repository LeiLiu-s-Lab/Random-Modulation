%% MIMO multipath channel
% Authors: 
% Yuhao Chi (yhchi@xidian.edu.cn)
% Yao Ge (yao.ge@ntu.edu.sg)
function H_TD_m = Get_channel(M, N, Nr, Ns, rho, delta_f, fs, fs_N, P, index_D, dop, beta)
    for i = 1 : P
        A(:, :, i) = relatedh(Nr, Ns, rho);
    end
    H_TD_ce = cell(Ns, Nr);
    for ns = 1 : Ns
        for nr = 1 : Nr
            [~, g_m_0, ~, ~, ~, ~, ~] = getchannel_related...
                (M, N, delta_f, fs, fs_N, P, index_D, dop, beta, A(nr,ns,:));
            L = size(g_m_0, 2) - 1;         
            H_T = zeros(N*M, N*M);        % Time-domain channel matrix MN*MN 
            for aa = 1 : N*M
                temp = g_m_0(aa, :);
                temp1 = fliplr(temp);
                temp2 = [temp1(end), zeros(1,N*M-L-1), temp1(1:end-1)];
                H_T(aa,:) = circshift(temp2,aa-1,2);
            end
            H_TD_ce{nr,ns} = H_T;
        end
    end
    % MIMO time-domain channel, size(H_TD_m) = [M*N*Nr, M*N*Ns]
    H_TD_m = cell2mat(H_TD_ce);             
end

%%
function A = relatedh(Nr, Ns, rho)
    C_R = eye(Nr, Nr);
    for i = 1 : Nr
        for j = 1 : Nr
            if abs(i - j) < 30
                C_R(i, j) = rho^(abs(i - j));
            end
        end
    end
    C_R = sqrtm(C_R);
    C_T = eye(Ns, Ns);
    for i = 1 : Ns
        for j = 1 : Ns
            if abs(i - j) < 30
                C_T(i, j) = rho^(abs(i - j));
            end
        end
    end
    C_T = sqrtm(C_T);
    G = sqrt(1/2) * (randn(Nr, Ns) + randn(Nr, Ns)*1i);
    A = C_R * G * C_T;
end

%%
function [HH, g, vv, pdb, tau, NN, t]= getchannel_related(M, N, delta_f, fs, fs_N, P, ...
    index_D, dop, beta, h)  
    MN = M * N;
    tau_fix = [0 0.3 0.67 0.95 1.5] .* 1e-6;  % delay of the TU channel
    tau = tau_fix(1:P) + 3 * fs_N / fs;    
    pdb = h;
    if index_D
        theat = pi * (2*rand(1,P)-1);
        vv = dop * cos(theat);
    else
        vv = zeros(1, P);
    end
    NN = 0:1/fs:max(tau) + 3 * fs_N / fs;
    t = 0:1/fs:MN/fs - 1/fs;
    g = [];
    for mc = 1 : length(t)
        for i = 1 : length(NN)
            for pp = 1 : P
                hh(pp) = pdb(pp)*exp(1i*2*pi*vv(pp)*(t(mc)-NN(i)))*myrrc(beta,NN(i)-tau(pp),fs/fs_N,1);
            end
            hhh(i) = sum(hh);
        end
        g(mc,:) = hhh;
    end
    HH = zeros(N*M);
    for i = 1 : P
        mu_k(i) = round(vv(i)*N/delta_f);
        if vv(i)>0
            beta_k(i) = -(abs(mu_k(i)*delta_f/N)-abs(vv(i)))/delta_f*N;
        else 
            beta_k(i) = (abs(mu_k(i)*delta_f/N)-abs(vv(i)))/delta_f*N;
        end
    end
    for l = 0 : M-1
        for k = 0 : N-1
            for pp1 = 1 : length(NN)
                pp = pp1 - 1;
                for i = 1 : P
                    for q = 0 : N-1
                        Tep = pdb(i)*exp(1i*2*pi*(l-pp)*(mu_k(i)+beta_k(i))/M/N)*myrrc(beta,NN(pp1)-tau(i),fs/fs_N,1)*...
                            ((exp(-1i*2*pi*(-q-beta_k(i)))-1)/(exp(-1i*2*pi*(-q-beta_k(i))/N)-1))/N;
                        if l < pp
                            Tep = Tep*exp(-1i*2*pi*mod(k-mu_k(i)+q,N)/N);
                        end 
                        HH(k*M+l+1,mod(k-mu_k(i)+q,N)*M+mod(l-pp,M)+1) = ...
                        HH(k*M+l+1,mod(k-mu_k(i)+q,N)*M+mod(l-pp,M)+1) + Tep;
                    end
                end
            end
        end
    end
end

function b = myrrc(beta, t1, fs, sps)
    t = t1 * fs;
    if abs(1-(2*beta*t).^2)>sqrt(eps)
        b = sinc(t).*(cos(pi*beta*t))./(1-(2*beta*t).^2)/sps;
    else
        b = beta*sin(pi/(2*beta))/(2*sps);
    end
end
