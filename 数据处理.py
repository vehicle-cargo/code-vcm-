import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Sigmoid function
# 向上收敛函数
def func1(x, a, b):
    return a / (1 + np.exp(-b * x))

# 向下收敛函数
def func2(x, c, d):
    return c - (c / (1 + np.exp(-d * (x))))

# Generating sample data for x between 0 and 10
x = np.linspace(0, 10, 100)

# Generate y1 and y2 using the sigmoid function and adding noise
y1 = func1(x, 5, 1) + np.random.normal(0, 0.2, size=100)
y2 = func2(x, 1, 1) + np.random.normal(0, 0.02, size=100)

# Plotting original data
fig, ax1 = plt.subplots()
ax1.plot(x, y1, 'r.', label='original data 1')
ax2 = ax1.twinx()
ax2.plot(x, y2, 'b.', label='original data 2')
fig.tight_layout()
fig.legend(loc="upper left", bbox_to_anchor=(0,1), bbox_transform=ax1.transAxes)
plt.show()

# Curve fitting 拟合一个收敛函数
popt1, pcov1 = curve_fit(func1, x, y1, maxfev=5000)
popt2, pcov2 = curve_fit(func2, x, y2, maxfev=5000)

y1_fit = func1(x, *popt1)
y2_fit = func2(x, *popt2)

# 计算方差
residuals1 = y1 - func1(x, *popt1)
std_dev1 = np.std(residuals1)

residuals2 = y2 - func2(x, *popt2)
std_dev2 = np.std(residuals2)

y1_fit_up = y1_fit + std_dev1
y1_fit_down = y1_fit - std_dev1

y2_fit_up = y2_fit + std_dev2
y2_fit_down = y2_fit - std_dev2


# Plotting fitted curve with confidence intervals
fig, ax1 = plt.subplots()
ax1.plot(x, y1, 'r.', label='original data 1')
ax1.plot(x, y1_fit, 'r-', label='fitted curve y1')
ax1.fill_between(x, y1_fit_down, y1_fit_up, color='pink', alpha=0.5)
ax1.set_xlabel('X axis')
ax1.set_ylabel('Y1 axis', color='r')
ax1.tick_params('y', colors='r')
# Set the limit for the right y-axis (associated with ax2)
ax1.set_ylim(2.5, 6)

ax2 = ax1.twinx()
ax2.plot(x, y2, 'b.', label='original data 2')
ax2.plot(x, y2_fit, 'b-', label='fitted curve y2')
ax2.fill_between(x, y2_fit_down, y2_fit_up, color='blue', alpha=0.1)
ax2.set_ylabel('Y2 axis', color='b')
ax2.tick_params('y', colors='b')
# Set the limit for the right y-axis (associated with ax2)
# ax2.set_ylim(0.4, 1.4)

fig.tight_layout()
fig.legend(loc="upper left", bbox_to_anchor=(0,1), bbox_transform=ax1.transAxes)
plt.show()
