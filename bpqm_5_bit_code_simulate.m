% Simulate the Belief-Propagation with Quantum Messages (BPQM) algorithm on
% the 5-bit code discussed in https://arxiv.org/abs/2003.04356

% This script generates the plots in Fig. 14 (main results) of this paper

% Author: Narayanan Rengaswamy
% Date: April 9, 2020

clc
clear

% Set to turn off the coherent rotation introduced after measuring bit 1
coherent_rotation_off = 0;

% Number of iterations, i.e. random codeword transmissions, per data point
blocks = 1e3;  % Increase for accuracy of performance plots

% Parity-check matrix
H = [1 1 1 0 0; ...
     1 0 0 1 1];

% Generator matrix
Cgen = [0 0 0 1 1; ...
        0 1 1 0 0; ...
        1 0 1 0 1];

% Full codebook    
C = mod(de2bi((0:7)',3) * Cgen, 2);

n = size(H,2);

e0 = [1; 0]; % |0>
e1 = [0; 1]; % |1>
E = [e0, e1];
E0 = e0*e0';
E1 = e1*e1';
Had = (1/sqrt(2)) * [1 1; 1 -1];    % Hadamard gate
X = [0 1; 1 0];
Z = [1 0; 0 -1];
CNOT = kron(E0, eye(2)) + kron(E1, X);
CNOTrev = kron(eye(2), E0) + kron(X, E1);
Swap = CNOT * CNOTrev * CNOT;

% Outputs |\theta> and |-\theta> of the pure-state channel
pure_state = @(t,s) (cos(t/2) * e0 + (-1)^s * sin(t/2) * e1);
pure_density = @(t,s) (pure_state(t,s) * pure_state(t,s)');

% Construct the variable node unitary in https://arxiv.org/abs/1607.04833
a_p = @(th1,th2) ( (1/sqrt(2)) * (cos((th1-th2)/2) + cos((th1+th2)/2)) / sqrt(1 + cos(th1) * cos(th2)) );
a_m = @(th1,th2) ( (1/sqrt(2)) * (cos((th1-th2)/2) - cos((th1+th2)/2)) / sqrt(1 + cos(th1) * cos(th2)) );
b_p = @(th1,th2) ( (1/sqrt(2)) * (sin((th1+th2)/2) - sin((th1-th2)/2)) / sqrt(1 - cos(th1) * cos(th2)) );
b_m = @(th1,th2) ( (1/sqrt(2)) * (sin((th1+th2)/2) + sin((th1-th2)/2)) / sqrt(1 - cos(th1) * cos(th2)) );
U = @(th1,th2) ([ a_p(th1,th2), 0, 0, a_m(th1,th2) ; ...
                  a_m(th1,th2), 0, 0, -a_p(th1,th2) ; ...
                   0, b_p(th1,th2), b_m(th1,th2), 0 ; ...
                   0, b_m(th1,th2), -b_p(th1,th2), 0 ]);

% Pure-state channel with parameter theta
th = 0.01:0.02:0.49;
biterr = zeros(5,length(th));
blockerr = zeros(1,length(th));
biterr_BP = zeros(5,length(th));
blockerr_BP = zeros(1,length(th));
biterr_ML = zeros(5,length(th));
blockerr_ML = zeros(1,length(th));
for i = 1:length(th)
    theta = th(i)*pi;
    p = 0.5 * (1 - sqrt(1 - cos(theta).^2));
    llr0 = log2((1-p)/p);
    
    psi(:,1) = cos(theta/2) * e0 + sin(theta/2) * e1;  % |\theta>
    psi(:,2) = cos(theta/2) * e0 - sin(theta/2) * e1;  % |-\theta>
    
    % blocks = 1e3;  % Increase for accuracy of performance plots
    
    for iter = 1:blocks
        x = C(randi(2^(n-size(H,1))),:);
        %     x = zeros(1,n);
        
        Y = psi(:, x + 1);  % Channel output states
        
        y = mod(x + (rand(1,n) < p),2);
        
        % Block MAP <-> Block ML, assuming equal prior on all codewords
        dist_to_C = sum(mod(y + C,2), 2);
        prob_codeword = p.^dist_to_C .* (1 - p).^(n - dist_to_C);
        xhat_ML = C(find(prob_codeword == max(prob_codeword), 1, 'first'), :);
        
        % Classical BP
        L = ((-1).^y) * llr0;
        
        xhat_BP = cbp(H, L);
        
        biterr_BP(:,i) = biterr_BP(:,i) + (xhat_BP ~= x)';
        biterr_ML(:,i) = biterr_ML(:,i) + (xhat_ML ~= x)';
        blockerr_BP(i) = blockerr_BP(i) + any(xhat_BP ~= x);
        blockerr_ML(i) = blockerr_ML(i) + any(xhat_ML ~= x);
        
        % BPQM for bit 1
        Rho_a = 1;
        for j = 1:n
            Rho_a = kron(Rho_a, Y(:,j));
        end
        
        Rho_b = kron_multi({eye(2), CNOT, CNOT}) * Rho_a;

        Rho_c = kron_multi({eye(2), eye(2), Swap, eye(2)}) * Rho_b;
        
        theta0 = acos(2 * cos(theta)/(1 + (cos(theta))^2));
        theta1 = pi/2;
        Ujoint = kron_multi({eye(2), U(theta0,theta0), E0, E0}) + ...
            kron_multi({eye(2), U(theta0,theta1), E0, E1}) + ...
            kron_multi({eye(2), U(theta1,theta0), E1, E0}) + ...
            kron_multi({eye(2), U(theta1,theta1), E1, E1});
        
        Rho_d = Ujoint * Rho_c;
        
        theta00 = acos( (cos(theta0))^2 );
        Vjoint = kron_multi({U(theta,theta00), eye(2), E0, E0}) + ...
            kron_multi({U(theta,pi/2), eye(2), E0, E1}) + ...
            kron_multi({U(theta,pi/2), eye(2), E1, E0}) + ...
            kron_multi({U(theta,pi/2), eye(2), E1, E1});
        
        Rho_e = Vjoint * Rho_d;
        
        State_1_op = kron(Had, eye(16)) * Rho_e;
        
        prob1_0 = State_1_op' * kron(E0, eye(16)) * State_1_op;
        prob1_1 = 1 - prob1_0;
        xhat(1) = (rand(1) < prob1_1);  % Estimate bit 1 from measurement
        
        % Postprocessing with coherent rotation
        phi00 = acos( cos(theta) * cos(theta00) );
        
        Kp = (1/sqrt(2)) * [ (cos(phi00/2) + sin(phi00/2)) * e0 + (sin(phi00/2) - cos(phi00/2)) * e1, ...
            (cos(phi00/2) - sin(phi00/2)) * e0 + (sin(phi00/2) + cos(phi00/2)) * e1 ];
        
        Km = (1/sqrt(2)) * [ (sin(phi00/2) + cos(phi00/2)) * e0 + (cos(phi00/2) - sin(phi00/2)) * e1, ...
            (sin(phi00/2) - cos(phi00/2)) * e0 + (cos(phi00/2) + sin(phi00/2)) * e1 ];
        
        bit1val = xhat(1);
        
        if (bit1val == 0)
            K = Kp;
            State_1_op_meas = kron(Had * E0, eye(16)) * State_1_op/sqrt(prob1_0);
        else
            K = Km;
            State_1_op_meas = kron(Had * E1, eye(16)) * State_1_op/sqrt(prob1_1);
        end
        
        if (coherent_rotation_off)
            K = eye(2);
        end
        
        % Reverse BPQM operations for bit 1
        Rho_e_rev = (kron_multi({K, eye(4), kron(E0, E0)}) + ...
            kron_multi({eye(2), eye(4), eye(4) - kron(E0, E0)})) * State_1_op_meas;

        Rho_d_rev = Vjoint' * Rho_e_rev;
        
        Rho_c_rev = Ujoint' * Rho_d_rev;
        
        Rho_b_rev = kron_multi({eye(2), eye(2), Swap, eye(2)}) * Rho_c_rev;
        
        Rho_a_rev = kron_multi({eye(2), CNOT, CNOT}) * Rho_b_rev;
        

        % BPQM for bit 2
%         W = kron(Had, eye(2)) * U(theta,theta) * kron(Z^(xhat(1)), eye(2));
        W = kron(Had, eye(2)) * U((-1)^(bit1val) * theta, theta);
        State_2_4_op = kron_multi({eye(2), W, W}) * Rho_a_rev;
        
        prob2_0 = State_2_4_op' * kron_multi({eye(2), E0, eye(2), eye(2), eye(2)}) * State_2_4_op;
        %prob24_01 = State_2_4_op' * kron_multi({eye(2), E0, eye(2), E1, eye(2)}) * State_2_4_op;
        %prob24_10 = State_2_4_op' * kron_multi({eye(2), E1, eye(2), E0, eye(2)}) * State_2_4_op;
        %prob24_11 = State_2_4_op' * kron_multi({eye(2), E1, eye(2), E1, eye(2)}) * State_2_4_op;
        prob4_0 = State_2_4_op' * kron_multi({eye(2), eye(2), eye(2), E0, eye(2)}) * State_2_4_op;
        prob2_1 = 1 - prob2_0;
        prob4_1 = 1 - prob4_0;
        
        xhat(2) = mod((rand(1) < prob2_1) + bit1val, 2);
        xhat(4) = mod((rand(1) < prob4_1) + bit1val, 2);
        xhat(3) = mod(bit1val + xhat(2), 2);
        xhat(5) = mod(bit1val + xhat(4), 2);
        
        biterr(:,i) = biterr(:,i) + (xhat ~= x)';
        blockerr(i) = blockerr(i) + any(xhat ~= x);
        
        if (mod(iter, floor(blocks/10)) == 0)
            clc
            disp(th(i));
            disp(iter);
            1 - biterr(:,1:(i-1))/blocks
            1 - blockerr(1:(i-1))/blocks
        end
    end
end
biterr = biterr/blocks;
blockerr = blockerr/blocks;
biterr_BP = biterr_BP/blocks;
blockerr_BP = blockerr_BP/blocks;
biterr_ML = biterr_ML/blocks;
blockerr_ML = blockerr_ML/blocks;

% theta_full = th * pi;
theta_full = [th, 1 - th(1, end:-1:1)]*pi;
biterr = [biterr, biterr(:, end:-1:1)];
blockerr = [blockerr, blockerr(1, end:-1:1)];
biterr_BP = [biterr_BP, biterr_BP(:, end:-1:1)];
blockerr_BP = [blockerr_BP, blockerr_BP(1, end:-1:1)];
biterr_ML = [biterr_ML, biterr_ML(:, end:-1:1)];
blockerr_ML = [blockerr_ML, blockerr_ML(1, end:-1:1)];

figure;
plot(theta_full/pi, 1 - mean(biterr_BP,1)', '--b');
hold on
plot(theta_full/pi, 1 - blockerr_BP, '--r');
plot(theta_full/pi, 1 - blockerr_ML, '--k');

plot(theta_full/pi, 1 - biterr(1,:), 'xr');
plot(theta_full/pi, 1 - biterr(2,:), '-og');
plot(theta_full/pi, 1 - biterr(3,:), 'xb');
plot(theta_full/pi, 1 - blockerr, '-xk');
plot(theta_full/pi, 1 - mean(biterr,1)', '--ob');

p0 = (1 + cos(theta_full).^2)/2;
Psucc1_bpqm = 1 - (p0.^2 - sqrt(p0.^4 - (2*p0 - 1).^3))/2;
Psucc2_bpqm = 0.5 * (1 + sqrt(1 - cos(theta_full).^4));

plot(theta_full/pi, Psucc1_bpqm, '-r'); 
plot(theta_full/pi, Psucc2_bpqm, '-m'); 

legend('Sim: BP Avg. Bit Success Rate', 'Sim: BP Block Success Rate', 'Sim: Codeword ML Block Success Rate', ...
       'Sim: BPQM Bit 1 Success Rate', 'Sim: BPQM Bit 2,4 Success Rate', ...
       'Sim: BPQM Bit 3,5 Success Rate', 'Sim: BPQM Block Success Rate', ...
       'Sim: BPQM Avg. Bit Success Rate', 'Theory: BPQM Bit 1 Success Rate', ...
       'Theory: BPQM Bits 2-5 Success Rate');
grid on
ylim([0,1]);


% Plot Yuen-Kennedy-Lax (YKL) limit from collected data, along with
% performance of symbol-by-symbol Helstrom + BP, symbol-by-symbol Helstrom
% + ML, and BPQM
Nbar = -0.5 * log(cos(theta_full));

figure;
semilogx(Nbar, mean(biterr_BP,1)', '--b');
hold on
semilogx(Nbar, blockerr_BP, '--r');
semilogx(Nbar, blockerr_ML, '--k');

semilogx(Nbar, biterr(1,:), 'xr');
semilogx(Nbar, biterr(2,:), '-og');
semilogx(Nbar, biterr(3,:), 'xb');
semilogx(Nbar, blockerr, '-xk');
semilogx(Nbar, mean(biterr,1)', '--ob');

p0 = (1 + cos(theta_full).^2)/2;
Psucc1_bpqm = 1 - (p0.^2 - sqrt(p0.^4 - (2*p0 - 1).^3))/2;
Psucc2_bpqm = 0.5 * (1 + sqrt(1 - cos(theta_full).^4));

semilogx(Nbar, 1 - Psucc1_bpqm, '-r'); 
semilogx(Nbar, 1 - Psucc2_bpqm, '-m'); 

% Data for YKL limit computed using https://arxiv.org/abs/1507.04737
cwHelxp = struct2dataset(load('datax_nbar.mat'));
cwHelyp = struct2dataset(load('datay_perr_vs_nbar.mat'));
semilogx(cwHelxp,cwHelyp,'-.k','linewidth',1.5)

legend('Sim: BP Avg. Bit Error Rate', 'Sim: BP Block Error Rate', 'Sim: Codeword ML Block Error Rate', ...
       'Sim: BPQM Bit 1 Error Rate', 'Sim: BPQM Bit 2,4 Error Rate', ...
       'Sim: BPQM Bit 3,5 Error Rate', 'Sim: BPQM Block Error Rate', ...
       'Sim: BPQM Avg. Bit Error Rate', 'Theory: BPQM Bit 1 Error Rate', ...
       'Theory: BPQM Bits 2-5 Min. Error Rate', 'Theory: Codeword Helstrom Limit');
grid on
ylim([0,1]);
xlabel('Mean photon number / mode','Interpreter','latex');
ylabel('Bit / Block Error Probability','Interpreter','latex');



