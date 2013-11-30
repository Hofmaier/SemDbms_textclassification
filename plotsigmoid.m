
x = (-5:0.1:5)';
y = 1./(1+e.^-x);
plot(x,y);
xlabel("x");
ylabel("sigmoid(x)");
print -dpng sigmoid.png
