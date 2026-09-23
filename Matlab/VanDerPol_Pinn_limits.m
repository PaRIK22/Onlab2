clear; clc; close all;
mu = 1.5; 

% --- LIMIT TESZT 1: KEVESEBB ADATPONT
dt = 0.01;
t_meres = (0:dt:20)';
%x''- mu(t) * (1-x^2) * x' + x = 0
%x_1 -> pozíció, x_2 -> sebesség
% x' = x(2) -> sebesség
% x'' = (1.5 - 0.05*t) * (1 - x(1)^2) * x(2) - x(1)
%allapotter reprezentacio
ode_func = @(t, x) [x(2); (1.5 - 0.05*t) * (1 - x(1)^2) * x(2) - x(1)];
[~, x_adat] = ode45(ode_func, t_meres, [0.1; 0]); 
% csak az ellenőrzéshez zajmentes adatok
x_tiszta = x_adat(:, 1);

% --- LIMIT TESZT 2: ZAJ
zaj_szint = 0.1; 
x_zajos = x_tiszta + zaj_szint * randn(size(x_tiszta));

T_dl = dlarray(t_meres', 'CB'); 
X_cel_dl = dlarray(x_zajos', 'CB');

% --- LIMIT TESZT 3: NEURON SZÁMOK ---
neuron_szam = 20; 
layers = [
    featureInputLayer(1, 'Normalization', 'none') 
    fullyConnectedLayer(neuron_szam)                    
    tanhLayer                                     
    fullyConnectedLayer(neuron_szam)                    
    tanhLayer                                     
    fullyConnectedLayer(1)                        
];

net = dlnetwork(layers);
numEpochs = 5000;
trailingAvg = []; %gradiens mozgóátlag  
sqTrailingAvg = []; %gradiens négyzet mozgóátlag
learnRate = 0.005;  

for epoch = 1:numEpochs
    [loss, gradients] = dlfeval(@pinnLoss, net, T_dl, X_cel_dl); 
    
    [net, trailingAvg, sqTrailingAvg] = adamupdate(net, gradients, trailingAvg, sqTrailingAvg, epoch, learnRate);
    
    if mod(epoch, 500) == 0
        fprintf('Epoch %d/%d - Teljes Hiba: %.4f\n', epoch, numEpochs, extractdata(loss));
    end
end

x_nn_predikcio = extractdata(predict(net, T_dl)); 
x_nn_plot = x_nn_predikcio'; 
figure('Name', 'PINN', 'Position', [100, 100, 800, 400]);
plot(t_meres, x_zajos, 'k', 'LineWidth', 0.5); hold on;
plot(t_meres, x_tiszta, 'b', 'LineWidth', 2);
plot(t_meres, x_nn_plot, 'r--', 'LineWidth', 2);
title('Position (x): Noisy data vs. Ideal data vs. PINN');
xlabel('Time [s]'); ylabel('Position [x]');
legend('Noisy data', 'Ideal data', 'PINN');
grid on;

function [loss, gradients] = pinnLoss(net, t_dl, x_cel_dl)
    x_pred = forward(net, t_dl);
    
    loss_data = mse(x_pred, x_cel_dl);
    
    v_pred = dlgradient(sum(x_pred, 'all'), t_dl, 'EnableHigherDerivatives', true);
    a_pred = dlgradient(sum(v_pred, 'all'), t_dl, 'EnableHigherDerivatives', true);
    
    mu_eff = 1.5 - 0.05 * t_dl;
    a_fizika = a_pred - mu_eff .* (1 - x_pred.^2) .* v_pred + x_pred;
    % --- LIMIT TESZT 4: ROSSZ FIZIKAI MODELL ---
    % mu_eff_hibas = 3.0; 
    % a_fizika = a_pred - mu_eff_hibas .* (1 - x_pred.^2) .* v_pred + x_pred;
    
    loss_physics = mean(a_fizika.^2, 'all');
    
    lambda_phys = 0.1; 
    loss = loss_data + lambda_phys * loss_physics;
    
    gradients = dlgradient(loss, net.Learnables); 
end

