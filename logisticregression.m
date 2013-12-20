clear ; close all; clc

function g = sigmoid (z)
g = 1.0 ./ (1.0 + exp(-z));
end

function grad = gradient(theta, X, y)
m = length(y);
h_theta = sigmoid(X * theta);
grad = 1/m .* (X' * (h_theta - y));
end

function theta = gradientDescent(X, y, theta, alpha, num_iters)
%gradientDescent fuehrt einen Gradientenabstieg 
%aus und gibt den Vektor Theta zurueck.
%   theta = gradientDescent(x, y, theta, alpha, num_iters) 
%   aktuallisiert theta indem 
%   der Gradientenabstieg num_iters Schritte mit 
%   der Lernrate alpha ausfuehrt.

m = length(y); % Anzahl Trainingsbeispiele
for iter = 1:num_iters
  grad = gradient(theta, X, y);
  theta = theta - alpha * grad;
end
end

function plotDecisionBoundary(theta, X, y)
figure; hold on;

pos = find(y==1);
neg = find(y==0);
plot(X(pos,1), X(pos,2), 'k+', 'LineWidth', 2, 'MarkerSize', 7);
plot(X(neg,1), X(neg, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7);

xlabel('X1')
ylabel('X2')

plot_x = [min(X(:,2))-2,  max(X(:,2))+2];
plot_y = (-1./theta(3)).*(theta(2).*plot_x + theta(1));
plot(plot_x, plot_y)
legend('Klasse 1', 'Klasse 0', 'Decision Boundary')
axis([30, 100, 30, 100])
hold off;
end

%% Lade Daten
data = load('ex2data1.txt');
X = data(:, [1, 2]); y = data(:, 3);

% Fuege der Matrix mit den Trainingsbeipielen
% eine Kolonne mit 1'en hinzu.
X = [ones(size(X,1), 1) X];

% Initialisiere Theta
initial_theta = zeros(size(X,2), 1);
theta = initial_theta;

%% Optimiere theta mit Gradientenabstieg

% Parameter fuer Gradientenabstieg
alpha = 0.01;
num_iters = 400000;

% starte gradientenabstieg
theta = gradientDescent(X, y, theta, alpha, num_iters);

plotDecisionBoundary(theta, X, y);
