# Mini Project 1: Additive Secret Sharing-based Distributed Average Consensus
# Paper: https://vbn.aau.dk/ws/portalfiles/portal/317462183/additive_eusipco.pdf
import numpy as np
from network import	Topology, create_network
from typing import Dict, List
import random
import matplotlib.pyplot as plt
import copy

def dp_obfuscation(topology: Topology, epsilon: float, distribution: str = 'laplace'):
	
    original_true_avg = sum(n.data for n in topology.nodes) / len(topology.nodes)

    delta = 1.0  # Sensitivity
    for node in topology.nodes:
        if distribution == 'laplace':
            noise = np.random.laplace(0, delta / epsilon)
        elif distribution == 'gaussian':
            noise = np.random.normal(0, delta / epsilon)
        else:
            noise = np.random.uniform(-(delta/epsilon), (delta/epsilon))
        
        node.data += noise
        
    return original_true_avg

def additive_sharing(topology: Topology, share_floor: int = 1, share_ceiling: int = 1000) -> float:
	
	# Tracks the specific share sent from sender_id to receiver_id
	sent_shares: Dict[int, Dict[int, int]] = {i.id: {} for i in topology.nodes} 
	
	# Tracks the sum of shares received by each node from all its neighbors
	received_sums: Dict[int, int] = {i.id: 0 for i in topology.nodes}         

	# Calculate the true original sum for verification
	initial_sum_a = sum(n.data for n in topology.nodes)

	# Draw Random Shares
	for node in topology.nodes:
		node_id = node.id
		
		# Draw a random share for each neighbor
		for neighbor_id in node.neighbors.keys():
			# Draw a random share between floor and ceiling
			share = random.randint(share_floor, share_ceiling) 
			sent_shares[node_id][neighbor_id] = share

	# Exchange and accumulate received sums
	for _, shares_to_send in sent_shares.items():
		for receiver_id, share in shares_to_send.items():
			# Accumulates the received numbers
			received_sums[receiver_id] += share

	# Adjust initial value
	print("\n--- Performing Additive Secret Sharing (Setup) ---")
	
	for node in topology.nodes:
		node_id = node.id
		original_a = node.data
		
		# Sum of numbers sent by this node
		sent_sum = sum(sent_shares[node_id].values())
		
		# Sum of numbers received by this node
		received_sum = received_sums[node_id] 
		
		# Adjust the initial value
		shared_s = original_a - sent_sum + received_sum
		
		# Update node data
		node.data = shared_s 
		
		print(f"Node {node_id}: a_i={original_a}, Sent Sum={sent_sum}, Received Sum={received_sum} -> New State s_i={shared_s}")
		
	final_sum_s = sum(n.data for n in topology.nodes)
	print(f"\nVerification: Original Sum (a_i): {initial_sum_a}. Final Sum (s_i): {final_sum_s}.")
	# Return true avg
	return initial_sum_a / len(topology.nodes)

def run_synchronous_consensus(topology: 'Topology', true_average: float, max_iter: int = 10000, tolerance: float = 0.001) -> List[float]:
	"""
	Synchronous approach: each node updates based on the weighted average of itself and all neighbors using the static W matrix.
	"""
	print("\nRunning Synchronous Consensus")
	
	N = len(topology.nodes)
	W = topology.w_matrix
	
	x_vector = [float(n.data) for n in topology.nodes]
	error_history = [] 
	initial_error = max(abs(val - true_average) for val in x_vector)
	error_history.append(initial_error)
	for t in range(1, max_iter + 1):
		
		new_x_vector = [0.0] * N
		
		for i in range(N):
			update_value = 0.0
			for j in range(N):
				# Calculate the weighted sum using the W matrix
				update_value += W[i][j] * x_vector[j] 
			new_x_vector[i] = update_value
			
		x_vector = new_x_vector
		
		# Check convergence
		max_error = max(abs(val - true_average) for val in x_vector)
		error_history.append(max_error)
		#print(max_error)
		
		if max_error < tolerance:
			print(f"Synchronous consensus converged after {t} iterations. Final max error: {max_error:.5f}")
			# Return error_history for plotting the convergence graph
			break
	return error_history

def run_asynchronous_consensus(topology: 'Topology', true_average: float, max_iter: int = 10000, threshold: float = 0.001) -> List[float]:
	"""
	Asynchronous approach: In each step, a random edge is activated, and the two connected nodes average their values.
	"""
	print("\nRunning Asynchronous Consensus")

	# The list of edges (pairs of node IDs) to choose from
	all_edges = topology.edges
	
	x_vector = [float(n.data) for n in topology.nodes]
	error_history = [] 
	initial_error = max(abs(val - true_average) for val in x_vector)
	error_history.append(initial_error)
	for t in range(1, max_iter + 1):
		
		# Randomly activate one edge (i, j)
		id1, id2 = random.choice(all_edges)
		
		# Convert 1-based IDs to 0-based list indices
		i, j = id1 - 1, id2 - 1
		
		# Get current values
		xi_t = x_vector[i]
		xj_t = x_vector[j]
		
		# Compute the average
		new_average = (xi_t + xj_t) / 2
		
		# Update the values (both nodes update to the new average)
		x_vector[i] = new_average
		x_vector[j] = new_average
		
		max_error = max(abs(val - true_average) for val in x_vector)
		error_history.append(max_error)
		#print(max_error)
		if max_error < threshold:
			print(f"Asynchronous consensus converged after {t} iterations. Final max error: {max_error:.5f}")
			# Return error_history for plotting the convergence graph
			break
	return error_history

def plot_convergence(sync_errors: List[float], async_errors: List[float], filename: str = 'convergence_plot.png'):
	"""
	Plots the convergence of synchronous and asynchronous consensus algorithms.
	"""
	print(f"\nGenerating Plot: {filename}")
	
	sync_iterations = range(1, len(sync_errors) + 1)
	
	async_iterations = range(1, len(async_errors) + 1)

	plt.figure(figsize=(10, 6))
	
	# Plot Synchronous Error
	plt.plot(sync_iterations, sync_errors, label='Synchronous', color='blue')
	
	# Plot Asynchronous Error
	plt.plot(async_iterations, async_errors, label='Asynchronous', color='red', linestyle='--')

	plt.xlabel('Iterations')
	plt.ylabel('Error')
	plt.yscale('log') 
	
	plt.title('Convergence Comparison')
	plt.legend()
	plt.grid(True, which="both", ls="--", linewidth=0.5)
	plt.savefig(filename)
	plt.close()
	
	print(f"Plot saved successfully as {filename}")

if __name__ == '__main__':
	num_nodes = 30
	alpha = 1 / num_nodes
	nodes, edges = create_network('ring', num_nodes, 20, 30)
	topology = Topology(nodes, edges, alpha)
	true_avg = additive_sharing(topology=topology)
	#true_avg = dp_obfuscation(topology=topology, epsilon=10)
	sync_errors = run_synchronous_consensus(topology=topology, true_average=true_avg)
	async_errors = run_asynchronous_consensus(topology=topology, true_average=true_avg)

	plot_convergence(sync_errors, async_errors, filename='convergence_plot.png')
	'''
    num_nodes = 10
    alpha = 1 / num_nodes
    
    # 1. Create the base network
    original_nodes, edges = create_network('star', num_nodes, 20, 30)
    base_topology = Topology(original_nodes, edges, alpha)

    # Dictionary to store error results for plotting
    all_results = {}

    # --- Scenario 1: Additive Secret Sharing ---
    # We use deepcopy to ensure we don't permanently mess up the base_topology
    topo_ss = copy.deepcopy(base_topology)
    true_avg = additive_sharing(topology=topo_ss)
    # We'll just track the Synchronous error for the main comparison
    all_results['Secret Sharing (Exact)'] = run_synchronous_consensus(topo_ss, true_avg)

    # --- Scenario 2: Differential Privacy (Varying Epsilon) ---
    # Higher epsilon = Lower noise (More accurate)
    # Lower epsilon  = Higher noise (More private)
    epsilons = [0.1, 1.0, 10.0]
    
    for eps in epsilons:
        topo_dp = copy.deepcopy(base_topology)
        # DP returns the true avg of the original data for error calculation
        true_avg = dp_obfuscation(topo_dp, epsilon=eps, distribution='laplace')
        label = f'DP (eps={eps})'
        all_results[label] = run_synchronous_consensus(topo_dp, true_avg)

    # --- 3. Custom Plotting for Multiple Lines ---
    plt.figure(figsize=(12, 7))
    for label, errors in all_results.items():
        plt.plot(range(len(errors)), errors, label=label)

    plt.xlabel('Iterations')
    plt.ylabel('Error (Difference from True Average)')
    plt.yscale('log')
    plt.title('Consensus Convergence: Secret Sharing vs. Differential Privacy')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    
    filename = 'comparison_plot.png'
    plt.savefig(filename)
    plt.show()
    print(f"\nComprehensive plot saved as {filename}")
	'''