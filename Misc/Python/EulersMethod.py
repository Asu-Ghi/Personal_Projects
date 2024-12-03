import numpy as np
from matplotlib import pyplot as plt


def function(t, y):
    return t**2 * np.exp(-2*t) - 2*y

def exact_sol(t):
    return (t**3 / 3 + 1) * np.exp(-2*t)

# max for t
def forward_euler(h, max):
    n_steps = int(max / h)

    t_vals = np.linspace(0, max, n_steps + 1)
    y_vals = np.zeros(n_steps + 1)

    # Initial condition
    y_vals[0] = 1

    # Forward Euler method
    for n in range(n_steps):
        y_vals[n+1] = y_vals[n] + h * function(t_vals[n], y_vals[n])
    
    return t_vals, y_vals

def get_error(y_exact, y_euler):
    return np.abs(y_exact - y_euler)


# Part A, Forward Euler 0<= i <= 10, h = 0.1
h = 0.1
t_max = 1
t_vals_a, y_vals_a = forward_euler(h, t_max)
y_exact_a = exact_sol(t_vals_a)
error_a = get_error(y_exact_a, y_vals_a)

# Part (b) - Forward Euler 0<= i <= 20, h = 0.05
h_b = 0.05
t_vals_b, y_vals_b = forward_euler(h_b, t_max)
y_exact_b = exact_sol(t_vals_b)
error_b = get_error(y_exact_b, y_vals_b)

# Get errors at t = 1 (last in the arrays)
t1_error_a = error_a[-1]
t1_error_b = error_b[-1]

# Plot the results
plt.figure(figsize=(12, 6))

# Part A
plt.subplot(1, 2, 1)
plt.plot(t_vals_a, y_vals_a, label='Euler (h=0.1)', color='blue')
plt.plot(t_vals_a, y_exact_a, label='Exact Solution', color='red', linestyle='dashed')
plt.title('Forward Euler Method with h = 0.1')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.legend()
plt.text(0.1, 0.0125, f"Error at y(1) (PART A): {t1_error_a:.5f}", fontsize=12, color='red')

# Part B
plt.subplot(1, 2, 2)
plt.plot(t_vals_b, y_vals_b, label='Euler (h=0.05)', color='blue')
plt.plot(t_vals_b, y_exact_b, label='Exact Solution', color='red', linestyle='dashed')
plt.title('Forward Euler Method with h = 0.05')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.legend()
plt.text(.1, 0.0125, f"Error at y(1)(PART B): {t1_error_b:.5f}", fontsize=12, color='red')

save_path = 'part_a&b.png' 
plt.savefig(save_path, format='png') 

# Get error at t = 1 (last in the array)


