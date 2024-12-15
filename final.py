import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class RNAVelocitySimulation:
    def __init__(self, n_cells=600, n_genes=15, beta=1.2):
        self.n_cells = n_cells
        self.n_genes = n_genes
        self.beta = beta
        self.T = 2 * np.log(12)
        
    def generate_parameters(self):
        """Generate parameters for simulation"""
        np.random.seed(42)
        alphas = np.random.uniform(15, 35, self.n_genes)
        gammas = np.random.uniform(1.0, 2.5, self.n_genes)
        switch_times = np.random.uniform(0.5, 1.5, self.n_genes)
        return alphas, gammas, switch_times
    
    def dynamics_on_stage(self, t, alpha, beta, u0=0, s0=0):
        """Compute dynamics for on-stage"""
        u = u0 * np.exp(-beta * t) + alpha / beta * (1 - np.exp(-beta * t))
        s = (s0 * np.exp(-beta * t) + 
             alpha / beta * (1 - np.exp(-beta * t)) - 
             (alpha - beta * u0) * t * np.exp(-beta * t))
        return u, s
    
    def dynamics_off_stage(self, t, u_switch, s_switch, beta, gamma, switch_time):
        """Compute dynamics for off-stage"""
        u = u_switch * np.exp(-beta * (t - switch_time))
        s = (s_switch * np.exp(-gamma * (t - switch_time)) - 
             beta * u_switch / (gamma - beta) * 
             (np.exp(-gamma * (t - switch_time)) - np.exp(-beta * (t - switch_time))))
        return u, s
    
    def generate_data(self):
        """Generate simulated RNA velocity data"""
        alphas, gammas, switch_times = self.generate_parameters()
        
        # Sampling times
        times = np.random.uniform(0, self.T, (self.n_cells, self.n_genes))
        
        # Initialize data matrices
        u_obs = np.zeros((self.n_cells, self.n_genes))
        s_obs = np.zeros((self.n_cells, self.n_genes))
        
        for g in range(self.n_genes):
            for c in range(self.n_cells):
                t = times[c, g]
                
                # Determine stage and compute dynamics
                if t <= switch_times[g]:
                    u, s = self.dynamics_on_stage(t, alphas[g], self.beta)
                else:
                    u_switch, s_switch = self.dynamics_on_stage(switch_times[g], alphas[g], self.beta)
                    u, s = self.dynamics_off_stage(t, u_switch, s_switch, 
                                                   self.beta, gammas[g], switch_times[g])
                
                # Add Gaussian noise
                u_obs[c, g] = u + np.random.normal(0, 0.25)
                s_obs[c, g] = s + np.random.normal(0, 0.25)
        
        return u_obs, s_obs, alphas, gammas, switch_times

def objective_function(params, u_obs, s_obs, times, beta, switch_times):
    """Objective function: Negative log-likelihood"""
    alphas = params[:len(switch_times)]
    gammas = params[len(switch_times):2*len(switch_times)]
    
    log_likelihood = 0
    n_cells, n_genes = u_obs.shape
    
    for g in range(n_genes):
        for c in range(n_cells):
            t = times[c, g]
            if t <= switch_times[g]:
                u, s = simulator.dynamics_on_stage(t, alphas[g], beta)
            else:
                u_switch, s_switch = simulator.dynamics_on_stage(switch_times[g], alphas[g], beta)
                u, s = simulator.dynamics_off_stage(t, u_switch, s_switch, beta, gammas[g], switch_times[g])
            
            diff_u = u - u_obs[c, g]
            diff_s = s - s_obs[c, g]
            log_likelihood -= (diff_u**2 + diff_s**2) / (2 * 0.25**2)
    
    return -log_likelihood

def m_step_nelder_mead(u_obs, s_obs, times, beta, switch_times, initial_params):
    result = minimize(objective_function, initial_params, args=(u_obs, s_obs, times, beta, switch_times), method='Nelder-Mead', options={'xatol': 1e-8, 'disp': True})
    return result.x

# --- Simulation and Fitting ---
simulator = RNAVelocitySimulation()
u_obs, s_obs, true_alphas, true_gammas, true_switch_times = simulator.generate_data()
times = np.random.uniform(0, simulator.T, (simulator.n_cells, simulator.n_genes))

initial_params = np.random.uniform(15, 35, simulator.n_genes).tolist() + np.random.uniform(1.0, 2.5, simulator.n_genes).tolist()
optimized_params = m_step_nelder_mead(u_obs, s_obs, times, simulator.beta, true_switch_times, initial_params)

optimized_alphas = optimized_params[:simulator.n_genes]
optimized_gammas = optimized_params[simulator.n_genes:]

# --- Visualization: Comparison of True vs Estimated RNA Velocities ---
n_cells_sample = 50  # Sample size for plotting
cell_indices = np.random.choice(range(simulator.n_cells), n_cells_sample, replace=False)

true_u = u_obs[cell_indices]
true_s = s_obs[cell_indices]

# Recompute u and s using optimized parameters
estimated_u = np.zeros_like(true_u)
estimated_s = np.zeros_like(true_s)

for i, c in enumerate(cell_indices):
    for g in range(simulator.n_genes):
        t = times[c, g]
        if t <= true_switch_times[g]:
            u, s = simulator.dynamics_on_stage(t, optimized_alphas[g], simulator.beta)
        else:
            u_switch, s_switch = simulator.dynamics_on_stage(true_switch_times[g], optimized_alphas[g], simulator.beta)
            u, s = simulator.dynamics_off_stage(t, u_switch, s_switch, simulator.beta, optimized_gammas[g], true_switch_times[g])
        estimated_u[i, g] = u
        estimated_s[i, g] = s

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(true_u.flatten(), true_s.flatten(), label="True RNA Velocity", alpha=0.5, color='blue')
plt.scatter(estimated_u.flatten(), estimated_s.flatten(), label="Estimated RNA Velocity", alpha=0.5, color='red')
plt.plot([0, max(true_u.flatten())], [0, max(true_s.flatten())], 'k--', lw=1)  # Reference line
plt.xlabel("Unspliced RNA (u)")
plt.ylabel("Spliced RNA (s)")
plt.title("Comparison of True vs Estimated RNA Velocities")
plt.legend()
plt.show()

