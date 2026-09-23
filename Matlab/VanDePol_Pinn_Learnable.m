clear; clc; close all;

%% 1. ADATGENERÁLÁS (Referencia szimuláció)
dt = 0.01;
t_meres = (0:dt:25)';

% Valós fizikai paraméterek (ezeket szeretnénk visszanyerni: a=1.5, b=0.05)
ode_func = @(t, x) [x(2); (1.5 - 0.05*t) * (1 - x(1)^2) * x(2) - x(1)];
[~, x_adat] = ode45(ode_func, t_meres, [0.1; 0]); 
x_tiszta = x_adat(:, 1);

% Zaj hozzáadása a valós mérés szimulálására
zaj_szint = 0.1; 
x_zajos = x_tiszta + zaj_szint * randn(size(x_tiszta));

% dlarray formátum a Deep Learning Toolboxhoz
T_dl = dlarray(t_meres', 'CB'); 
X_cel_dl = dlarray(x_zajos', 'CB');

%% 2. ARCHITEKTÚRA ÉS TANULHATÓ FIZIKAI PARAMÉTEREK
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

% --- ÚJ: Tanulandó fizikai paraméterek inicializálása kezdeti tippel ---
% Cél: a = 1.5, b = 0.05
param_a = dlarray(1.0);   % Kezdeti becslés a-ra
param_b = dlarray(0.01);  % Kezdeti becslés b-re

%% 3. OPTIMALIZÁLÓ BEÁLLÍTÁSAI
numEpochs = 20000;         % Inverz feladathoz több lépés javasolt
learnRate_net = 0.005;    % Hálózat tanulási rátája
learnRate_a = 0.003;   % 'a' maradhat gyors, hogy hamar elérje a másfelet
learnRate_b = 0.0005;  % 'b' jóval finomabb, nehogy elszaladjon felfelé

% Adam belső memóriái a hálózathoz
trailingAvg_net = [];   
sqTrailingAvg_net = []; 

% Adam belső memóriái az 'a' paraméterhez
trailingAvg_a = [];
sqTrailingAvg_a = [];

% Adam belső memóriái a 'b' paraméterhez
trailingAvg_b = [];
sqTrailingAvg_b = [];

%% 4. TANÍTÁSI CIKLUS (TRAINING LOOP)
fprintf('Tanítás indítása (Inverz feladat: pálya + fizikai paraméterek becslése)...\n');

% Tömbök a tanítási görbék naplózásához
loss_history = zeros(numEpochs, 1);
a_history = zeros(numEpochs, 1);
b_history = zeros(numEpochs, 1);

for epoch = 1:numEpochs
    % Gradiens és hiba számítás a módosított veszteségfüggvénnyel
    [loss, grad_net, grad_a, grad_b] = dlfeval(@pinnLoss, net, param_a, param_b, T_dl, X_cel_dl); 
    
    % 1. Hálózat súlyainak frissítése
    [net, trailingAvg_net, sqTrailingAvg_net] = adamupdate(...
        net, grad_net, trailingAvg_net, sqTrailingAvg_net, epoch, learnRate_net);
    
    [param_a, trailingAvg_a, sqTrailingAvg_a] = adamupdate(param_a, grad_a, trailingAvg_a, sqTrailingAvg_a, epoch, learnRate_a);
    [param_b, trailingAvg_b, sqTrailingAvg_b] = adamupdate(param_b, grad_b, trailingAvg_b, sqTrailingAvg_b, epoch, learnRate_b);
    
    % --- ÉRTÉKEK MENTÉSE AZ ÁBRÁKHOZ ---
    loss_history(epoch) = extractdata(loss);
    a_history(epoch) = extractdata(param_a);
    b_history(epoch) = extractdata(param_b);

    % Állapot kiírása 500 lépésenként
    if mod(epoch, 500) == 0
        a_akt = extractdata(param_a);
        b_akt = extractdata(param_b);
        fprintf('Epoch %4d/%d | Hiba: %.5f | Becsült a: %.4f (Valós: 1.5) | Becsült b: %.4f (Valós: 0.05)\n', ...
            epoch, numEpochs, extractdata(loss), a_akt, b_akt);
    end
end

%% 5. KIÉRTÉKELÉS ÉS VIZUALIZÁCIÓ
x_nn_predikcio = extractdata(predict(net, T_dl)); 
x_nn_plot = x_nn_predikcio'; 

figure('Name', 'PINN Inverz Eredmények', 'Position', [100, 100, 900, 500]);
plot(t_meres, x_zajos, 'k', 'LineWidth', 0.5); hold on;
plot(t_meres, x_tiszta, 'b', 'LineWidth', 2);
plot(t_meres, x_nn_plot, 'r--', 'LineWidth', 2);

a_vegleges = extractdata(param_a);
b_vegleges = extractdata(param_b);

title(sprintf('Inverz PINN illesztés | Becsült: a = %.3f, b = %.4f | Valós: a = 1.50, b = 0.05', ...
    a_vegleges, b_vegleges));
xlabel('Time [s]'); ylabel('Position [x]');
legend('Noisy data', 'Ideal data', 'PINN prediction');
grid on;

%% --- PARAMÉTEREK ÉS HIBA KONVERGENCIÁJÁNAK PLOTOLÁSA ---
figure('Name', 'Inverz Paraméter Konvergencia', 'Position', [150, 150, 950, 450]);

% 1. Alábra: A veszteségfüggvény csökkenése (Logaritmikus skálán)
subplot(1, 2, 1);
semilogy(1:numEpochs, loss_history, 'k-', 'LineWidth', 1.5);
title('Veszteség alakulása (Loss)');
xlabel('Epoch'); ylabel('Loss (log skála)');
grid on;

% 2. Alábra: Az 'a' és 'b' paraméterek konvergenciája
subplot(1, 2, 2);
plot(1:numEpochs, a_history, 'b-', 'LineWidth', 2); hold on;
yline(1.5, 'b--', 'Valós a = 1.5', 'LineWidth', 1.5, 'LabelHorizontalAlignment', 'left');

plot(1:numEpochs, b_history, 'r-', 'LineWidth', 2);
yline(0.05, 'r--', 'Valós b = 0.05', 'LineWidth', 1.5, 'LabelHorizontalAlignment', 'left');

title('Fizikai paraméterek konvergenciája');
xlabel('Epoch'); ylabel('Paraméter érték');
legend('Becsült a(t)', 'Elméleti a', 'Becsült b(t)', 'Elméleti b', 'Location', 'best');
grid on;

%% 6. MÓDOSÍTOTT VESZTESÉGFÜGGVÉNY
function [loss, grad_net, grad_a, grad_b] = pinnLoss(net, param_a, param_b, t_dl, x_cel_dl)
    % Előrecsatolás
    x_pred = forward(net, t_dl);
    
    % Adatveszteség (Data loss)
    loss_data = mse(x_pred, x_cel_dl);
    
    % Automatikus differenciálás a sebességhez és gyorsuláshoz
    v_pred = dlgradient(sum(x_pred, 'all'), t_dl, 'EnableHigherDerivatives', true);
    a_pred = dlgradient(sum(v_pred, 'all'), t_dl, 'EnableHigherDerivatives', true);
    
    % Fizikai reziduál a tanulandó paraméterekkel: mu_eff = a - b * t
    mu_eff = param_a - param_b .* t_dl;
    a_fizika = a_pred - mu_eff .* (1 - x_pred.^2) .* v_pred + x_pred;
    
    % Fizikai veszteség
    loss_physics = mean(a_fizika.^2, 'all');
    
    % Teljes hibafüggvény súlyozása
    lambda_phys = 1.0; 
    loss = loss_data + lambda_phys * loss_physics;
    
    % Gradiens a hálózati súlyok és a két skalár paraméter szerint
    [grad_net, grad_a, grad_b] = dlgradient(loss, net.Learnables, param_a, param_b); 
end