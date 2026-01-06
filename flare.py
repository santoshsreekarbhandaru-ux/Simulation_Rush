import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("flare_data.csv")

import numpy as np

def model(t,A,tau,omega):
    return A*np.exp(t)*(1-np.tanh(2*(t-tau)))*np.sin(omega*t)
y_guess=model(data["t"],A=2.0,tau=4.0,omega=9)
def log_likelihood(y_true,y_guess):
    sigma=0.2*np.abs(y_true)
    sigma[sigma == 0] = 1e-6 
    return -np.sum((y_true-y_guess)**2/sigma**2)
#err=error(data["s"],y_guess)

best_omega=9.5
best_A=1.6
best_tau=7.0
best_y=model(data["t"],best_A,best_tau,best_omega)
best_logL=log_likelihood(data["s"],best_y)

#storing parameter chains
A_chain = []
tau_chain=[]
omega_chain=[]

for i in range(3000):
    omega_try=best_omega + np.random.normal(0,0.1)
    tau_try = best_tau + np.random.normal(0, 0.1)
    A_try = best_A + np.random.normal(0, 0.02)

    if not (0 < A_try < 2 and 1 < tau_try < 10 and 1 < omega_try < 20):
        continue

    y_try=model(data["t"],A_try,tau_try,omega_try)
    logL_try=log_likelihood(data["s"],y_try)
    
    if logL_try > best_logL:
        accept = True
    else:
        accept_prob = np.exp(logL_try - best_logL)
        accept = np.random.rand() < accept_prob

    if accept:
        best_logL = logL_try
        best_A = A_try
        best_tau = tau_try
        best_omega = omega_try

    A_chain.append(best_A)
    tau_chain.append(best_tau)
    omega_chain.append(best_omega)

final_y=model(data["t"],best_A,best_tau,best_omega)

#trace plot for A
plt.figure()
plt.plot(A_chain)
plt.xlabel("Iteration")
plt.ylabel("A")
plt.title("Trace plot for A")
plt.show()
#trace plote for tau
plt.figure()
plt.plot(tau_chain)
plt.xlabel("Iteration")
plt.ylabel("Tau")
plt.title("Trace plot for Tau")
plt.show()
#trace plot for omega
plt.figure()
plt.plot(omega_chain)
plt.xlabel("Iteration")
plt.ylabel("Omega")
plt.title("Trace plot for Omega")
plt.show()

#histograms

#histogram for A
burn_in=int(0.2*len(A_chain))
plt.figure()
plt.hist(A_chain[burn_in:], bins=40, density=True)
plt.xlabel("A")
plt.ylabel("Probability Density")
plt.title("Posterior distribution of A")
plt.show()

#Histogram for tau
plt.figure()
plt.hist(tau_chain[burn_in:], bins=40, density=True)
plt.xlabel("Tau")
plt.ylabel("Probability Density")
plt.title("Posterior distribution of Tau")
plt.show()

#histogram for omega
plt.figure()
plt.hist(omega_chain[burn_in:], bins=40, density=True)
plt.xlabel("Omega")
plt.ylabel("Probability Density")
plt.title("Posterior distribution of Omega")
plt.show()



#best vs guess plot
plt.plot(data["t"],data["s"],'.',markersize=2,label="Raw data")
plt.plot(data["t"],final_y,'r',label="Model guess")
plt.xlabel("Time")
plt.ylabel("Signal")
plt.title("Raw solar flare")
plt.legend()
plt.show()
print("Best omega:", best_omega)

print("Best A =", best_A)
print("Best tau =", best_tau)
print("Best error:", best_logL)