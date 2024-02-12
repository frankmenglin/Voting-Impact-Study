from itertools import product
import numpy as np

#This function computes Banzhaf Power Index
#Given vote count of each person/party, estimates their actual impact
def banzhaf_power_index(votes):
    total_votes = sum(votes)
    majority = total_votes / 2
    power = [0] * len(votes)
    
    # All voting combination
    for combination in product([0, 1], repeat=len(votes)):
        sum_combination = sum(vote * comb for vote, comb in zip(votes, combination))
        for i, voter in enumerate(combination):
            if voter == 1:
                # Is this vote critical?
                if sum_combination - votes[i] <= majority < sum_combination:
                    power[i] += 1
    
    total_power = sum(power)
    if total_power == 0:
        return [0] * len(votes)
    
    # Banzhaf Power Index
    return [p / total_power for p in power]


def is_valid_correlation_matrix(matrix):
    # Check if the matrix is square
    if matrix.shape[0] != matrix.shape[1]:
        return False

    # Check if the matrix is symmetric
    if not np.allclose(matrix, matrix.T):
        return False

    # Check if the matrix is positive semi-definite
    if np.any(np.linalg.eigvals(matrix) < 0):
        return False

    return True

def generate_correlated_normals(correlation_matrix):
    # Generate multivariate normal with given correlation
    mean = np.zeros(len(correlation_matrix))
    return np.random.multivariate_normal(mean, correlation_matrix)

def vote_outcome(random_values, barriers, votes, majority_count, switch_vote=None):
    # Determine the voting outcome based on barriers, random values, and vote counts
    individual_votes = [(rv < barrier) for rv, barrier in zip(random_values, barriers)]
    if switch_vote is not None:
        individual_votes[switch_vote] = not individual_votes[switch_vote]  # Switch the specified vote

    total_yes_votes = sum(vote_count if vote else 0 for vote, vote_count in zip(individual_votes, votes))
    return total_yes_votes >= majority_count

#This function extends the idea of Banzhaf Power Index, to the situation where votes could be correlated.
#Given vote count of each person/party, and their voting probability/correlation, estimates their actual impact
def run_simulation(votes, barriers, correlation_matrix, majority_count, trials = 10000):
    # Convert lists to numpy arrays for easier manipulation
    votes = np.array(votes)
    barriers = np.array(barriers)
    correlation_matrix = np.array(correlation_matrix)

    # Check if the correlation matrix is valid
    if not is_valid_correlation_matrix(correlation_matrix):
        raise ValueError("Invalid correlation matrix")
    
    impact_counts = np.zeros(len(votes))
    for _ in range(trials):
        random_values = generate_correlated_normals(correlation_matrix)
        initial_outcome = vote_outcome(random_values, barriers, votes, majority_count)

        for i in range(len(votes)):
            # Check if changing this vote would have changed the outcome
            new_outcome = vote_outcome(random_values, barriers, votes, majority_count, switch_vote=i)
            if new_outcome != initial_outcome:
                impact_counts[i] += 1
    return impact_counts/trials

# Test case

"""
votes = [5,10,15,20,22,28]
print("The vote counts are,")
print(votes)
banzhaf_index = banzhaf_power_index(votes)
print("Their corresponding banzhaf index are ")
print(banzhaf_index)
"""

try:
    votes = [52,51,8,2] #2024 Taiwan legislator counts of KMT, DPP, TPP, independent (no party)
    barrier = [0, 0, 0, 0] #Equal chance to vote yes or no
    correlation_matrix = [[1, -0.8, 0, 0.5], [-0.8, 1, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]] #KMT,DPP usually against each other. 2 independents are favored KMT
    majority = 56.5 #Half of total vote count
    
    print("Simulate the situation where votes are correlated.")
    print("Vote count of each party is,")
    print(votes)
    print("Then we simulate votes by std. Gaussian Random Variable, barrier represents minimum value for the person votes no.")
    print("Barrier vector is,")
    print(barrier)
    print("Correlation of these random variables are given by the matrix below")
    print(correlation_matrix)
    print("Majority (minimum yes vote count to be considered overall yes vote outcount) is,")
    print(majority)
    impacts = run_simulation(votes, barrier, correlation_matrix, majority, trials = 100000)
    print("Each person's impact is,")
    print(impacts)
except ValueError as e:
    print("Error:", e)

#simulation_results = run_simulation(votes, barrier, correlation_matrix, majority)
#print(simulation_results)

