import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# --- 1. Define the Environment and Constraints ---
START = np.array([0.0, 0.0])
GOAL = np.array([10.0, 0.0])
N_NODES = 15                 # Number of discrete poses in our band
OBSTACLE = np.array([5.0, 0.1]) # Obstacle placed directly in the path
MIN_SAFE_DIST = 2.0          # How far the robot wants to stay from the obstacle

# Weights (gamma_k) to balance our competing forces
WEIGHT_TENSION = 1.0   # Pulls the band tight (shortest path)
WEIGHT_OBSTACLE = 20.0 # Pushes the band away from dangers

def teb_cost_function(internal_nodes_flat):
    """
    This is the mathematical heart of the planner.
    It calculates the total 'cost' or 'badness' of a given trajectory.
    """
    # Reshape the flat array back into (N, 2) coordinates
    internal_nodes = internal_nodes_flat.reshape((-1, 2))
    
    # Reconstruct the full band (Start -> Internal Nodes -> Goal)
    full_band = np.vstack((START, internal_nodes, GOAL))
    
    total_cost = 0.0
    
    # --- Penalty 1: Tension / Smoothness ---
    # We want consecutive nodes to be close together to minimize total distance
    for i in range(len(full_band) - 1):
        dist_between_nodes = np.linalg.norm(full_band[i+1] - full_band[i])
        total_cost += WEIGHT_TENSION * (dist_between_nodes ** 2)
        
    # --- Penalty 2: Obstacle Repulsion ---
    # If a node gets too close to the obstacle, apply a massive penalty
    for i in range(len(full_band)):
        dist_to_obs = np.linalg.norm(full_band[i] - OBSTACLE)
        if dist_to_obs < MIN_SAFE_DIST:
            # Quadratic penalty that gets worse the closer it gets
            total_cost += WEIGHT_OBSTACLE * ((MIN_SAFE_DIST - dist_to_obs) ** 2)
            
    return total_cost

def main():
    # 2. Initial Guess: A dumb straight line from Start to Goal
    x_init = np.linspace(START[0], GOAL[0], N_NODES)
    y_init = np.linspace(START[1], GOAL[1], N_NODES)
    initial_band = np.vstack((x_init, y_init)).T
    
    # We only optimize the internal nodes (Start and Goal are fixed)
    internal_nodes_guess = initial_band[1:-1].flatten()

    print("Optimizing trajectory...")
    
    # 3. The Optimizer (Python's equivalent to g2o)
    # We use BFGS, a gradient-based optimization algorithm
    result = minimize(
        teb_cost_function, 
        internal_nodes_guess, 
        method='BFGS',
        options={'disp': True}
    )

    # 4. Extract the optimized path
    optimized_internal_nodes = result.x.reshape((-1, 2))
    optimized_band = np.vstack((START, optimized_internal_nodes, GOAL))

    # --- 5. Visualization ---
    plt.figure(figsize=(10, 5))
    
    # Plot the obstacle and its safety radius
    obs_circle = plt.Circle(OBSTACLE, MIN_SAFE_DIST, color='red', alpha=0.2, label='Safety Zone')
    plt.gca().add_patch(obs_circle)
    plt.plot(OBSTACLE[0], OBSTACLE[1], 'ro', label='Obstacle Core')
    
    # Plot Initial and Optimized bands
    plt.plot(initial_band[:, 0], initial_band[:, 1], 'k--', alpha=0.5, marker='x', label='Initial Guess (Collides)')
    plt.plot(optimized_band[:, 0], optimized_band[:, 1], 'b-', linewidth=2, marker='o', label='Optimized Elastic Band')
    
    plt.plot(START[0], START[1], 'gs', markersize=10, label='Start')
    plt.plot(GOAL[0], GOAL[1], 'g*', markersize=12, label='Goal')
    
    plt.title("Simplified Elastic Band Optimization")
    plt.xlabel("X Position"); plt.ylabel("Y Position")
    plt.axis('equal'); plt.grid(True); plt.legend()
    plt.show()

if __name__ == "__main__":
    main()