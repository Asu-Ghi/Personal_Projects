import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Function to solve ODE using odeint
def system_of_odes(y, t, A):
    return np.dot(A, y)

# Function to solve ODE analytically using eigenvalues and eigenvectors
def analytical_solution(A, y0, t):
    # Find eigenvalues and eigenvectors of matrix A
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    # Initialize the solution array
    y_t = np.zeros((len(t), len(y0)))

    for i, eigenvalue in enumerate(eigenvalues):
        eigenvector = eigenvectors[:, i]
        
        coefficient = np.dot(np.linalg.inv(eigenvectors), y0)[i]
        
        if np.iscomplex(eigenvalue):
            real_part = np.real(eigenvalue)
            imag_part = np.imag(eigenvalue)
            
            cos_term = np.outer(np.real(eigenvector), np.cos(imag_part * t))
            sin_term = np.outer(np.imag(eigenvector), np.sin(imag_part * t))
            
            y_t += coefficient * np.real(cos_term + sin_term)  # Take real part
        else:
            y_t += coefficient * np.outer(eigenvector, np.exp(eigenvalue * t))

    return y_t.T  # Transpose to get the shape (len(t), len(y0))

# Main function to run the script
def solve_ode(A, y0, t):
    # Numerical solution using odeint
    sol_numerical = odeint(system_of_odes, y0, t, args=(A,))
    
    # Analytical solution using eigenvalues and eigenvectors
    sol_analytical = analytical_solution(A, y0, t)
    
    # Plotting the results
    plt.figure(figsize=(10, 6))
    
    for i in range(len(y0)):
        plt.plot(t, sol_numerical[:, i], label=f'Numerical Solution y_{i+1}(t)', linestyle='--')
        plt.plot(t, sol_analytical[:, i], label=f'Analytical Solution y_{i+1}(t)', linestyle='-')
    
    plt.xlabel('Time (t)')
    plt.ylabel('y(t)')
    plt.title('Comparison of Numerical and Analytical Solutions')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example matrix A and initial conditions
A = np.array([[2, -5], [1, -2]])  # Example matrix A
y0 = np.array([5, 2])  # Initial conditions y(0) = [5, 2]
t = np.linspace(0, 10, 100)  # Time interval from t=0 to t=10

# Call the solver
solve_ode(A, y0, t)
