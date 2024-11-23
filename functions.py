import math
import random
import json
import numpy as np
import time
import pickle
from itertools import combinations_with_replacement, combinations
from typing import List, Generator, Tuple
import networkx as nx
import plotly.graph_objects as go
import plotly.subplots as sp

def powerset(iterable, size=1):
    "powerset([1,2,3], 2) --> (1,2) (1,3) (2,3)"
    return [[i for i in item] for item in combinations(iterable, size)]
  
def get_v_star(T: int) -> np.ndarray:
    """
    Generate a set of vectors V* of length T, where each vector is a cyclic permutation of an initial vector.

    The initial vector 'u' is defined as:
    - u[0] = -1
    - u[-1] = 1
    - all other elements are 0

    Parameters:
    T (int): Length of the vectors.

    Returns:
    np.ndarray: An array of shape (T, T), where each row is a vector in V*.
    """
    # Create an initial vector 'u' of zeros with length 'T'
    u = np.zeros(T, dtype=np.int64)
    # Set the first element of vector 'u' to -1 and the last element to 1
    u[0] = -1
    u[-1] = 1
    # Initialize the list 'v_star' with the initial vector 'u'
    v_star = [u.copy()]

    # Generate shifted versions of 'u' by rotating it T-1 times
    for i in range(T - 1):
        # Rotate 'u' by moving the last element to the front
        u = np.roll(u, 1)
        # Append the updated vector 'u' to the list 'v_star'
        v_star.append(u.copy())

    # Return 'v_star' as an array of shape (T, T)
    return np.array(v_star)

def generate_all_schedules(N: int, T: int) -> List[List[int]]:
    """
    Generate all possible schedules of N patients over T time slots.

    Each schedule is represented as a list of length T, where each element
    indicates the number of patients scheduled in that time slot.

    Parameters:
    N (int): Total number of patients.
    T (int): Total number of time slots.

    Returns:
    List[List[int]]: A list containing all possible schedules.
    """
    def generate(current_schedule: List[int], remaining_patients: int, remaining_slots: int):
        """
        Recursive helper function to build schedules.

        Parameters:
        current_schedule (List[int]): The current partial schedule being built.
        remaining_patients (int): The number of patients left to schedule.
        remaining_slots (int): The number of time slots left to fill.
        """
        if remaining_slots == 0:
            if remaining_patients == 0:
                schedules.append(current_schedule)
            return

        # Try all possible numbers of patients for the current slot
        for i in range(remaining_patients + 1):
            generate(current_schedule + [i], remaining_patients - i, remaining_slots - 1)

    # Calculate the total number of possible schedules
    pop_size = math.comb(N + T - 1, N)
    print(f"Number of possible schedules: {pop_size}")
    schedules = []
    generate([], N, T)
    return schedules

def serialize_schedules(N: int, T: int) -> None:
    """
    Generate all possible schedules of N patients over T time slots and serialize them to a file.

    The schedules are stored in a pickle file, with one schedule per line.

    Parameters:
    N (int): Total number of patients.
    T (int): Total number of time slots.

    Returns:
    None
    """
    file_path = f"experiments/n{N}t{T}.pickle"

    # Start timing
    start_time = time.time()

    # Open a file for writing serialized schedules
    with open(file_path, 'wb') as f:
        # Generate all combinations with replacement
        for comb in combinations_with_replacement(range(T), N):
            # Count occurrences to get the schedule
            schedule = np.bincount(comb, minlength=T).tolist()
            # Serialize the schedule
            pickle.dump(schedule, f)

    # End timing
    end_time = time.time()

    print(f"Number of possible schedules: {math.comb(N + T - 1, N)}")
    print(f"Time taken: {end_time - start_time} seconds")
    print(f"Schedules are serialized and saved to {file_path}")

def load_schedules(file_path: str) -> List[List[int]]:
    """
    Load schedules from a pickle file.

    Parameters:
    file_path (str): Path to the pickle file containing the serialized schedules.

    Returns:
    List[List[int]]: The list of schedules loaded from the file.
    """
    # Start timing
    start_time = time.time()

    schedules = []
    with open(file_path, 'rb') as f:
        while True:
            try:
                # Load each schedule and append to the list
                schedules.append(pickle.load(f))
            except EOFError:
                # End of file reached
                break

    # End timing
    end_time = time.time()
    print(f"Loading time: {end_time - start_time} seconds")
    return schedules

def generate_random_schedule(N: int, T: int) -> List[int]:
    """
    Generate a random schedule for N patients over T time slots.

    Each patient is randomly assigned to one of the T time slots.

    Parameters:
    N (int): Total number of patients.
    T (int): Total number of time slots.

    Returns:
    List[int]: A schedule represented as a list of length T, where each element
               indicates the number of patients scheduled in that time slot.
    """
    # Randomly assign each patient to a time slot
    slots = random.choices(range(T), k=N)
    schedule = [0] * T
    for slot in slots:
        schedule[slot] += 1
    return schedule

def random_combination_with_replacement(T: int, N: int, num_samples: int) -> List[List[int]]:
    """
    Generate random samples from the set of combinations with replacement without generating the entire population.

    Parameters:
    T (int): The range of elements to choose from, i.e., the maximum value plus one.
    N (int): The number of elements in each combination.
    num_samples (int): The number of random samples to generate.

    Returns:
    List[List[int]]: A list containing the randomly generated combinations.
    """
    def index_to_combination(index: int, T: int, N: int) -> List[int]:
        """
        Convert a lexicographic index to a combination with replacement.

        Parameters:
        index (int): The lexicographic index of the combination.
        T (int): The range of elements to choose from, i.e., the maximum value plus one.
        N (int): The number of elements in each combination.

        Returns:
        List[int]: The combination corresponding to the given index.
        """
        combination = []
        current = index
        for i in range(N):
            for j in range(T):
                combs = math.comb(N - i + T - j - 2, T - j - 1)
                if current < combs:
                    combination.append(j)
                    break
                current -= combs
        return combination

    # Calculate total number of combinations
    total_combinations = math.comb(N + T - 1, N)
    print(f"Total number of combinations: {total_combinations:,}")

    schedules = []
    for _ in range(num_samples):
        # Randomly select an index
        random_index = random.randint(0, total_combinations - 1)
        # Convert index to combination
        sample = index_to_combination(random_index, T, N)
        # Convert combination to schedule
        schedule = np.bincount(sample, minlength=T).tolist()
        schedules.append(schedule)

    return schedules
  
def create_random_schedules(T, N, num_schedules):
  schedules = []
  for _ in range(num_schedules):
    sample = random.choices(range(T), k = N)
    schedule = np.bincount(sample, minlength=T).tolist()
    schedules.append(schedule)
  return(schedules)

def build_welch_bailey_schedule(N, T):
    """
    Build a schedule based on the Welch and Bailey (1952) heuristic.

    Parameters:
    N (int): Number of patients to be scheduled.
    T (int): Number of time intervals in the schedule.

    Returns:
    list: A schedule of length T where each item represents the number of patients scheduled
          at the corresponding time interval.
    """
    # Initialize the schedule with zeros
    schedule = [0] * T

    # Schedule the first two patients at the beginning
    schedule[0] = 2
    remaining_patients = N - 2

    # Distribute patients in the middle time slots with gaps
    for t in range(1, T - 1):
        if remaining_patients <= 0:
            break
        if t % 2 == 1:  # Create gaps (only schedule patients at odd time slots)
            schedule[t] = 1
            remaining_patients -= 1

    # Push any remaining patients to the last time slot
    schedule[-1] += remaining_patients

    return schedule

def create_neighbors_list(s: List[int]) -> (List[int], List[int]):
    """
    Create a neighbor schedule from the given schedule s.

    The neighbor is generated by adding or subtracting patients in time slots
    according to certain rules defined by the V* vectors.

    Parameters:
    s (List[int]): A schedule represented as a list of integers with length T and sum N.

    Returns:
    Tuple[List[int], List[int]]: A pair of schedules (s, s_p), where s_p is a neighbor of s.
    """
    T = len(s)
    # Create a set of vectors V* of length T
    v_star = get_v_star(T)

    # Choose a random integer i between 1 and T-1 with probability proportional to C(T, i)
    weights = [math.comb(T, i) for i in range(1, T)]
    i = random.choices(range(1, T), weights=weights)[0]

    # Create a list l of all subsets of t with length i
    l = list(combinations(range(T), i))

    # Choose a random subset j from l
    j = random.choice(l)

    # Select vectors from V* with indices in j
    v_j = [v_star[idx] for idx in j]

    # Sum the selected vectors and add to the schedule s
    s_p = s.copy()
    for v in v_j:
        # Add the vector v to the current schedule
        s_p_temp = [int(x + y) for x, y in zip(s_p, v)]
        # Check that no time slot has negative patients
        if np.all(np.array(s_p_temp) >= 0):
            s_p = s_p_temp

    return s, s_p

def create_neighbors_list_single_swap(S: List[List[int]]) -> List[Tuple[List[int], List[int]]]:
    """
    For each schedule in S, create a neighbor by swapping one patient from one time slot to another.

    Parameters:
    S (List[List[int]]): A list of schedules.

    Returns:
    List[Tuple[List[int], List[int]]]: A list of pairs of schedules, where each pair consists of the original schedule and its neighbor.
    """
    neighbors_list = []

    # For each schedule in the list
    for s in S:
        # Choose a random time slot i
        i = random.choice(range(len(s)))
        # Find time slots with at least one patient that are not i
        J = [index for index, element in enumerate(s) if element > 0 and index != i]

        if not J:
            # No valid time slots to swap from
            continue

        # Choose a random time slot j to swap from
        j = random.choice(J)

        # Create a copy of s to modify
        s_pair = s.copy()
        # Swap one patient from time slot j to time slot i
        s_pair[i] = s[i] + 1
        s_pair[j] = s[j] - 1

        neighbors_list.append((s, s_pair))

    return neighbors_list

def calculate_objective(schedule: List[int], s: List[float], d: int, q: float) -> Tuple[float, float]:
    """
    Calculate the objective value based on the given schedule and parameters.

    This function adjusts the service times distribution for no-shows, calculates 
    the waiting times for all patients in the schedule, sums the expected 
    waiting times, and calculates the spillover time for the last interval (overtime).

    Parameters:
    schedule (List[int]): A list representing the number of patients scheduled in each time slot.
    s (List[float]): Service times probability distribution.
    d (int): Duration threshold or maximum allowed service time per slot.
    q (float): No-show probability.

    Returns:
    Tuple[float, float]: 
        - ewt (float): The sum of expected waiting times.
        - esp (float): The expected spillover time (overtime).
    """
    # Adjust the service time distribution for no-shows
    s = service_time_with_no_shows(s, q)
    # Initialize the service process (probability distribution of waiting times)
    sp = np.array([1], dtype=np.float64)
    wt_list = []
    ewt = 0  # Expected waiting time
    for x in schedule:
        if x == 0:
            # No patients in this time slot
            wt_temp = [np.array(sp)]
            wt_list.append([])
            sp = []
            sp.append(np.sum(wt_temp[-1][:d + 1]))
            sp.extend(wt_temp[-1][d + 1:])
        else:
            # Patients are scheduled in this time slot
            wt_temp = [np.array(sp)]
            # Add expected waiting time for the first patient
            ewt += np.dot(range(len(sp)), sp)
            # For each additional patient
            for i in range(x - 1):
                # Convolve the waiting time with the service time distribution
                wt = np.convolve(wt_temp[i], s)
                wt_temp.append(wt)
                # Add expected waiting time
                ewt += np.dot(range(len(wt)), wt)
            wt_list.append(wt_temp)
            # Update the service process for the next time slot
            sp = []
            convolved = np.convolve(wt_temp[-1], s)
            sp.append(np.sum(convolved[:d + 1]))
            sp.extend(convolved[d + 1:])
    # Calculate expected spillover time
    esp = np.dot(range(len(sp)), sp)
    return ewt, esp
  
def calculate_objective_serv_time_lookup(schedule: List[int], d: int, q: float, convolutions: dict) -> Tuple[float, float]:
    """
    Calculate the objective value based on the given schedule and parameters using precomputed convolutions.

    This function uses precomputed convolutions of the service time distribution,
    starting from the 1-fold convolution (key 1) which contains the adjusted service time distribution.

    Parameters:
    schedule (List[int]): A list representing the number of patients scheduled in each time slot.
    d (int): Duration threshold or maximum allowed service time per slot.
    q (float): No-show probability.
    convolutions (dict): Precomputed convolutions of the service time distribution, with key 1 containing the adjusted service time distribution.

    Returns:
    Tuple[float, float]: 
        - ewt (float): The sum of expected waiting times.
        - esp (float): The expected spillover time (overtime).
    """
    sp = np.array([1], dtype=np.float64)  # Initial service process (no waiting time)
    ewt = 0  # Total expected waiting time

    for x in schedule:
        if x == 0:
            # No patients in this time slot
            # Adjust sp for the duration d (service process moves ahead)
            sp_new = []
            sp_new.append(np.sum(sp[:d + 1]))
            sp_new.extend(sp[d + 1:])
            sp = np.array(sp_new)
        else:
            # Patients are scheduled in this time slot
            wt_temp = [sp.copy()]
            # Add expected waiting time for the first patient
            ewt += np.dot(range(len(sp)), sp)
            # For each additional patient
            for i in range(1, x):
                # The waiting time distribution for the ith patient is the convolution
                # of the previous patient's waiting time with s (adjusted service time distribution)
                conv_s = convolutions.get(1)  # Adjusted service time distribution
                wt = np.convolve(wt_temp[i - 1], conv_s)
                wt_temp.append(wt)
                ewt += np.dot(range(len(wt)), wt)
            # Update sp for the next time slot
            conv_s = convolutions.get(1)  # Adjusted service time distribution
            sp = np.convolve(wt_temp[-1], conv_s)
            # Adjust sp for duration d
            sp_new = []
            sp_new.append(np.sum(sp[:d + 1]))
            sp_new.extend(sp[d + 1:])
            sp = np.array(sp_new)
    # Expected spillover time
    esp = np.dot(range(len(sp)), sp)
    return ewt, esp

def service_time_with_no_shows(s: List[float], q: float) -> List[float]:
    """
    Adjust a distribution of service times for no-shows.

    The adjusted service time distribution accounts for the probability q of no-shows.

    Parameters:
    s (List[float]): The original service time probability distribution.
    q (float): The no-show probability.

    Returns:
    List[float]: The adjusted service time distribution.
    """
    # Adjust the service times by multiplying with (1 - q)
    s_adj = [(1 - q) * float(si) for si in s]
    # Add the no-show probability to the zero service time
    s_adj[0] = s_adj[0] + q
    return s_adj

def calculate_ambiguousness(y_pred_proba: np.ndarray) -> np.ndarray:
    """
    Calculate the ambiguousness (entropy) for each sample's predicted probabilities.

    The ambiguousness is calculated as the entropy of the predicted class probabilities.

    Parameters:
    y_pred_proba (np.ndarray): Array of shape (n_samples, n_classes) with predicted probabilities for each class.

    Returns:
    np.ndarray: Array of ambiguousness (entropy) for each sample.
    """
    # Ensure probabilities are in numpy array
    y_pred_proba = np.array(y_pred_proba)

    # Define a small bias term to avoid log(0)
    epsilon = 1e-10

    # Add small bias term to probabilities that contain zeros
    y_pred_proba = np.clip(y_pred_proba, epsilon, 1 - epsilon)

    # Calculate ambiguousness (entropy) for each sample
    ambiguousness = -np.sum(y_pred_proba * np.log2(y_pred_proba), axis=1)

    return ambiguousness

def calculate_opaqueness(y_pred_proba: np.ndarray) -> float:
    """
    Calculate the opaqueness (entropy) for a set of normalized predicted objectives.

    Parameters:
    y_pred_proba (np.ndarray): Array of normalized predicted objectives.

    Returns:
    float: The opaqueness value.
    """
    # Ensure probabilities are in numpy array
    y_pred_proba = np.array(y_pred_proba)

    # Define a small bias term to avoid log(0)
    epsilon = 1e-10

    # Add small bias term to probabilities that contain zeros
    y_pred_proba = np.clip(y_pred_proba, epsilon, 1 - epsilon)

    # Calculate opaqueness (entropy) over the entire array
    opaqueness = -np.sum(y_pred_proba * np.log2(y_pred_proba))

    return opaqueness

def compare_json(json1: dict, json2: dict) -> dict:
    """
    Compare two JSON objects and return a dictionary with the differences.

    Parameters:
    json1 (dict): The first JSON object to compare.
    json2 (dict): The second JSON object to compare.

    Returns:
    dict: A dictionary showing the differences between the two JSON objects.
    """
    differences = {}

    # Check keys in json1
    for key in json1.keys():
        if key in json2:
            if json1[key] != json2[key]:
                differences[key] = {
                    "json1_value": json1[key],
                    "json2_value": json2[key]
                }
        else:
            differences[key] = {
                "json1_value": json1[key],
                "json2_value": "Key not found in json2"
            }

    # Check keys in json2 that are not in json1
    for key in json2.keys():
        if key not in json1:
            differences[key] = {
                "json1_value": "Key not found in json1",
                "json2_value": json2[key]
            }

    return differences

def compute_convolutions(probabilities: List[float], N: int, q: float = 0.0) -> dict:
    """
    Compute the k-fold convolution of a probability mass function (PMF) for k from 1 to N.

    The function adjusts the service time distribution for no-shows and computes
    convolutions for each k.

    Parameters:
    probabilities (List[float]): The PMF represented as a list where the index is the service time and the value is the probability.
    N (int): The maximum number of convolutions to compute.
    q (float): No-show probability (default 0.0).

    Returns:
    dict: A dictionary where keys are k and values are the convoluted service times (PMFs).
    """
    convolutions = {}
    # Adjust the service time distribution for no-shows
    result = probabilities.copy()
    result = service_time_with_no_shows(result, q)
    for k in range(1, N + 1):
        if k == 1:
            # First convolution is the adjusted service time distribution
            convolutions[k] = result
        else:
            # Convolve the result with the original probabilities
            result = np.convolve(result, probabilities)
            convolutions[k] = result
    return convolutions

def local_search(x, d, q, convolutions, w, v_star, size=2, echo=False):
    # Initialize the best solution found so far 'x_star' to the input vector 'x'
    x_star = np.array(x).flatten()  # Keep as 1D array

    # Calculate initial objectives and cost
    objectives_star = calculate_objective_serv_time_lookup(x_star, d, q, convolutions)
    c_star = w * objectives_star[0] + (1 - w) * objectives_star[1]

    # Set the value of 'T' to the length of the input vector 'x'
    T = len(x_star)

    # Outer loop for the number of patients to switch
    t = 1
    while t < size:
        if echo == True: print(f'Running local search {t}')

        # Generate the neighborhood of the current best solution 'x_star' with 't' patients switched
        ids_gen = powerset(range(T), t)
        neighborhood = get_neighborhood(x_star, v_star, ids_gen)
        if echo == True: print(f"Switching {t} patient(s). Size of neighborhood: {len(list(ids_gen))}")

        # Flag to track if a better solution is found
        found_better_solution = False

        for neighbor in neighborhood:
            # Calculate objectives for the neighbor
            objectives = calculate_objective_serv_time_lookup(neighbor, d, q, convolutions)
            cost = w * objectives[0] + (1 - w) * objectives[1]

            # Compare scalar costs
            if cost < c_star:
                x_star = neighbor
                c_star = cost
                if echo == True: print(f"Found better solution: {x_star}, cost: {c_star}")

                # Set the flag to restart the outer loop
                found_better_solution = True
                break  # Break out of the inner loop

        # If a better solution was found, restart the search from t = 1
        if found_better_solution:
            t = 1  # Restart search with t = 1
        else:
            t += 1  # Move to the next neighborhood size if no better solution was found

    # Return the best solution found 'x_star' and its cost
    return x_star, c_star
  
import numpy as np
from typing import List, Tuple

def local_search_w_intermediates(
    x: List[int],
    d: int,
    q: float,
    convolutions: any,  # Replace 'any' with the actual type if known
    w: float,
    v_star: any,        # Replace 'any' with the actual type if known
    size: int = 2,
    echo: bool = False
) -> Tuple[List[List[int]], List[float]]:
    """
    Performs a local search to find an optimal schedule, saving intermediate solutions.

    Parameters:
    - x (List[int]): Initial schedule.
    - d (int): Duration threshold.
    - q (float): No-show probability.
    - convolutions (any): Precomputed convolutions (replace 'any' with actual type).
    - w (float): Weight for the objective function.
    - v_star (any): Some precomputed value or structure (replace 'any' with actual type).
    - size (int): Maximum number of patients to switch in the neighborhood.
    - echo (bool): If True, prints debug information.

    Returns:
    - schedules (List[List[int]]): List containing the initial schedule and all intermediate improved schedules.
    - objectives (List[float]): List containing the corresponding objective values for each schedule.
    """
    
    # Initialize the best solution found so far 'x_star' to the input vector 'x'
    x_star = np.array(x).flatten()  # Ensure it's a 1D array

    # Calculate initial objectives and cost
    objectives_star = calculate_objective_serv_time_lookup(x_star, d, q, convolutions)
    c_star = w * objectives_star[0] + (1 - w) * objectives_star[1]

    # Initialize lists to store schedules and their corresponding objective values
    schedules: List[List[int]] = [x_star.tolist()]  # Start with the initial schedule
    objectives: List[float] = [c_star]             # Initial objective value

    # Set the value of 'T' to the length of the input vector 'x'
    T = len(x_star)

    # Outer loop for the number of patients to switch
    t = 1
    while t < size:
        if echo:
            print(f'Running local search with t = {t}')

        # Generate the neighborhood of the current best solution 'x_star' with 't' patients switched
        ids_gen = powerset(range(T), t)
        neighborhood = get_neighborhood(x_star, v_star, ids_gen)

        # To accurately count the neighborhood size, we need to regenerate the generator
        neighborhood_list = list(get_neighborhood(x_star, v_star, powerset(range(T), t)))
        if echo:
            print(f"Switching {t} patient(s). Size of neighborhood: {len(neighborhood_list)}")

        # Flag to track if a better solution is found
        found_better_solution = False

        for neighbor in neighborhood_list:
            # Calculate objectives for the neighbor
            objectives_neighbor = calculate_objective_serv_time_lookup(neighbor, d, q, convolutions)
            cost_neighbor = w * objectives_neighbor[0] + (1 - w) * objectives_neighbor[1]

            # Compare scalar costs
            if cost_neighbor < c_star:
                x_star = neighbor
                c_star = cost_neighbor
                schedules.append(x_star.tolist())
                objectives.append(c_star)

                if echo:
                    print(f"Found better solution: {x_star}, cost: {c_star}")

                # Set the flag to restart the outer loop
                found_better_solution = True
                break  # Break out of the inner loop to restart search

        # If a better solution was found, restart the search from t = 1
        if found_better_solution:
            t = 1  # Restart search with t = 1
        else:
            t += 1  # Move to the next neighborhood size if no better solution was found

    # Return the list of schedules and their corresponding objective values
    return schedules, objectives
  
def get_neighborhood(x, v_star, ids, verbose=False):
    x = np.array(x)
    p = 50
    if verbose:
        print(f"Printing every {p}th result")
    # Initialize the list 'neighborhood' to store the vectors in the neighborhood of 'x'
    neighborhood = []
    # Loop over all possible non-empty subsets of indices
    for i in range(len(ids)):
        # Initialize the vector 'neighbor' to store the sum of vectors in 'v_star' corresponding to the indices in 'ids[i]'
        neighbor = np.zeros(len(x), dtype=int)
        # Loop over all indices in 'ids[i]'
        for j in range(len(ids[i])):
            if verbose:
                print(f"v_star{[ids[i][j]]}: {v_star[ids[i][j]]}")
            # Add the vector in 'v_star' corresponding to the index 'ids[i][j]' to 'neighbor'
            neighbor += v_star[ids[i][j]]
        # Append the vector 'x' plus 'neighbor' to the list 'neighborhood'
        x_n = x + neighbor
        if i%p==0:
            if verbose:
                print(f"x, x', delta:\n{x},\n{x_n},\n{neighbor}\n----------------- ")
        neighborhood.append(x_n)
    
    # Convert the list 'neighborhood' into a NumPy array
    neighborhood = np.array(neighborhood)
    if verbose:
        print(f"Size of raw neighborhood: {len(neighborhood)}")
    # Create a mask for rows with negative values
    mask = ~np.any(neighborhood < 0, axis=1)
    # Filter out rows with negative values using the mask
    if verbose:
        print(f"filtered out: {len(neighborhood)-mask.sum()} schedules with negative values.")
    filtered_neighborhood = neighborhood[mask]
    if verbose:
        print(f"Size of filtered neighborhood: {len(filtered_neighborhood)}")
    return filtered_neighborhood
  
def create_schedule_network(N: int, T: int, s: List[float], d: int, q: float, w: float, echo: bool = False) -> go.Figure:
    """
    Creates and visualizes a network of schedules where each node represents an allocation
    of N patients across T time intervals. Edges connect schedules that differ by moving
    a single patient between any two time intervals. Node colors represent the objective value.
    Arrows are added from nodes with higher objective values to those with lower values.

    Parameters:
    - N (int): Total number of patients.
    - T (int): Number of time intervals.
    - s (List[float]): Service times probability distribution.
    - d (int): Duration threshold or maximum allowed service time per slot.
    - q (float): No-show probability.

    Returns:
    - fig (plotly.graph_objects.Figure): Interactive network graph.
    """

    # 1. Generate all possible schedules (compositions of N into T non-negative integers)
    def generate_compositions(n, t):
        if t == 1:
            yield (n,)
            return
        for i in range(n + 1):
            for comp in generate_compositions(n - i, t - 1):
                yield (i,) + comp

    schedules = list(generate_compositions(N, T))

    # 2. Initialize the directed graph
    G = nx.DiGraph()

    # Add all schedules as nodes
    for schedule in schedules:
        G.add_node(schedule)

    # 3. Define a helper function to find neighbors by moving one patient
    def find_neighbors(schedule):
        neighbors = []
        for i in range(T):
            for j in range(T):
                if i != j and schedule[i] > 0:
                    # Move one patient from interval i to interval j
                    new_schedule = list(schedule)
                    new_schedule[i] -= 1
                    new_schedule[j] += 1
                    neighbors.append(tuple(new_schedule))
        return neighbors

    # 4. Add edges based on single-interval shifts
    for schedule in schedules:
        neighbors = find_neighbors(schedule)
        for neighbor in neighbors:
            if neighbor in G.nodes():
                G.add_edge(schedule, neighbor)

    # 5. Assign positions to nodes
    if T == 3:
        # For T=3, arrange nodes in a triangular (simplex) layout
        positions = {}
        for schedule in G.nodes():
            a, b, c = schedule
            # Simplex coordinates for 3 variables
            # Convert to 2D coordinates for plotting
            x = a - c
            y = b - c
            positions[schedule] = (x, y)
    else:
        # For other T, use spring layout
        positions = nx.spring_layout(G, seed=42)  # Seed for reproducibility

    # Assign positions to graph nodes
    nx.set_node_attributes(G, positions, 'pos')

    # 6. Calculate objective values for each node
    # We'll use 'ewt' + 'esp' as a combined objective for coloring
    objective_values = []
    for schedule in G.nodes():
        ewt, esp = calculate_objective(list(schedule), s, d, q)
        objective = w*ewt/N + (1-w)*esp  # Combine the two objectives as needed
        if echo == True: print(f"Schedule: {schedule}, Objective: {objective:.2f}, Expected mean waiting time: {ewt:.2f}, Expected spillover time: {esp:.2f}")
        objective_values.append(objective)

    # Normalize objective values for color mapping
    min_obj = min(objective_values)
    max_obj = max(objective_values)
    if max_obj - min_obj == 0:
        normalized_values = [0.5 for _ in objective_values]
    else:
        normalized_values = [(val - min_obj) / (max_obj - min_obj) for val in objective_values]

    # Create a mapping from schedule to normalized objective value
    schedule_to_obj = {schedule: obj for schedule, obj in zip(G.nodes(), normalized_values)}

    # 7. Create edge traces (for visual reference, lines without direction)
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    # 8. Create node traces with updated hover info
    node_x = []
    node_y = []
    node_labels = []      # For node annotations (schedule)
    node_hovertext = []   # For hover info (objective value)
    node_colors = []

    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        node_x.append(x)
        node_y.append(y)
        # Node label shows the schedule
        node_labels.append(str(node))
        # Hover text shows the objective value
        node_hovertext.append(f"Objective: {schedule_to_obj[node]:.2f}")
        node_colors.append(schedule_to_obj[node])

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_labels,
        textposition="top center",
        hovertext=node_hovertext,
        hoverinfo='text',
        marker=dict(
            showscale=True,
            # Define colorscale from deep red to white
            colorscale=[
                [0.0, 'rgb(165,0,38)'],    # Deep Red
                [1.0, 'rgb(255,255,255)']  # White
            ],
            reversescale=True,
            color=node_colors,
            size=30,
            colorbar=dict(
                thickness=15,
                title=dict(
                    text='Objective Value',
                    side='right'
                ),
                xanchor='left',
            ),
            line_width=2
        )
    )

    # 9. Prepare arrow annotations based on directed edges with edge labels
    arrow_annotations = []
    for edge in G.edges():
        source, target = edge
        # Compare objective values to determine direction
        source_obj = schedule_to_obj[source]
        target_obj = schedule_to_obj[target]
        if source_obj > target_obj:
            # Arrow from source to target with negative difference
            diff = target_obj - source_obj  # Negative value
            x0, y0 = G.nodes[source]['pos']
            x1, y1 = G.nodes[target]['pos']
            # Calculate midpoint for label
            mid_x = (x0 + x1) / 2
            mid_y = (y0 + y1) / 2
            # Add arrow annotation
            arrow_annotations.append(dict(
                x=x1,
                y=y1,
                ax=x0,
                ay=y0,
                xref='x',
                yref='y',
                axref='x',
                ayref='y',
                showarrow=True,
                arrowhead=2,
                arrowsize=2.5,
                arrowwidth=1,
                arrowcolor="#000000",
                opacity=0.5
            ))
            # Add text annotation for the difference
            arrow_annotations.append(dict(
                x=mid_x,
                y=mid_y,
                text=f"{diff:.2f}",
                showarrow=False,
                xref='x',
                yref='y',
                font=dict(color='blue', size=10),
                align='center'
            ))
        elif target_obj > source_obj:
            # Arrow from target to source with negative difference
            diff = source_obj - target_obj  # Negative value
            x0, y0 = G.nodes[target]['pos']
            x1, y1 = G.nodes[source]['pos']
            # Calculate midpoint for label
            mid_x = (x0 + x1) / 2
            mid_y = (y0 + y1) / 2
            # Add arrow annotation
            arrow_annotations.append(dict(
                x=x1,
                y=y1,
                ax=x0,
                ay=y0,
                xref='x',
                yref='y',
                axref='x',
                ayref='y',
                showarrow=True,
                arrowhead=2,
                arrowsize=2.5,
                arrowwidth=1,
                arrowcolor="#000000",
                opacity=0.5
            ))
            # Add text annotation for the difference
            arrow_annotations.append(dict(
                x=mid_x,
                y=mid_y,
                text=f"{diff:.2f}",
                showarrow=False,
                xref='x',
                yref='y',
                font=dict(color='blue', size=10),
                align='center'
            ))
        # If equal, no arrow or label

    # 10. Assemble the figure with arrow annotations
    exp_s = sum([i * si for i, si in enumerate(s)])
    fig = go.Figure(data=[edge_trace, node_trace],
                 layout=go.Layout(
                    title=dict(
                        text=f"Network of Schedules with N={N} Patients, T={T} Time Intervals, Exp. service time = {exp_s:.2f} and w = {w}",
                        font=dict(size=16)
                    ),
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    annotations=arrow_annotations + [ dict(
                        text="Created with NetworkX and Plotly",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002 ) ],
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )

    return fig

def create_schedule_network_var_edges(N: int, T: int, s: List[float], d: int, q: float, w: float, echo: bool = False) -> go.Figure:
    """
    Creates and visualizes a network of schedules where each node represents an allocation
    of N patients across T time intervals. Edges connect schedules that differ by moving
    a single patient between any two time intervals. Node colors represent the objective value.
    Arrows are added from nodes with higher objective values to those with lower values.
    Edge lengths are proportional to the absolute difference in objective values.

    Parameters:
    - N (int): Total number of patients.
    - T (int): Number of time intervals.
    - s (List[float]): Service times probability distribution.
    - d (int): Duration threshold or maximum allowed service time per slot.
    - q (float): No-show probability.
    - w (float): Weighting factor for combining ewt and esp in objective calculation.
    - echo (bool): If True, prints schedule details and objective values.

    Returns:
    - fig (plotly.graph_objects.Figure): Interactive network graph.
    """

    # 1. Generate all possible schedules (compositions of N into T non-negative integers)
    def generate_compositions(n, t):
        if t == 1:
            yield (n,)
            return
        for i in range(n + 1):
            for comp in generate_compositions(n - i, t - 1):
                yield (i,) + comp

    schedules = list(generate_compositions(N, T))

    # 2. Initialize the directed graph
    G = nx.DiGraph()

    # Add all schedules as nodes
    for schedule in schedules:
        G.add_node(schedule)

    # 3. Define a helper function to find neighbors by moving one patient
    def find_neighbors(schedule):
        neighbors = []
        for i in range(T):
            for j in range(T):
                if i != j and schedule[i] > 0:
                    # Move one patient from interval i to interval j
                    new_schedule = list(schedule)
                    new_schedule[i] -= 1
                    new_schedule[j] += 1
                    neighbors.append(tuple(new_schedule))
        return neighbors

    # 4. Add edges based on single-interval shifts
    for schedule in schedules:
        neighbors = find_neighbors(schedule)
        for neighbor in neighbors:
            if neighbor in G.nodes():
                G.add_edge(schedule, neighbor)

    # 5. Assign positions to nodes based on objective values to make edge lengths proportional
    #    Arrange nodes along the x-axis based on their objective values
    #    Assign a small y-offset to spread nodes vertically and prevent overlap
    objective_dict = {}
    for schedule in G.nodes():
        ewt, esp = calculate_objective(list(schedule), s, d, q)
        objective = w * ewt / N + (1 - w) * esp  # Combine the two objectives as needed
        objective_dict[schedule] = objective
        if echo:
            print(f"Schedule: {schedule}, Objective: {objective:.2f}, Expected mean waiting time: {ewt:.2f}, Expected spillover time: {esp:.2f}")

    # Sort schedules by objective value
    sorted_schedules = sorted(G.nodes(), key=lambda x: objective_dict[x])

    # Assign positions
    positions = {}
    spacing = 1  # Base spacing between nodes
    y_offset = 0  # Initial y-offset
    y_increment = 0.2  # Increment for each node to prevent overlap

    for idx, schedule in enumerate(sorted_schedules):
        x = objective_dict[schedule] * spacing  # x-position proportional to objective value
        y = y_offset
        positions[schedule] = (x, y)
        y_offset += y_increment  # Increment y to spread nodes vertically

    # Assign positions to graph nodes
    nx.set_node_attributes(G, positions, 'pos')

    # 6. Calculate objective values for each node (already done in objective_dict)

    # Normalize objective values for color mapping
    min_obj = min(objective_dict.values())
    max_obj = max(objective_dict.values())
    if max_obj - min_obj == 0:
        normalized_values = [0.5 for _ in objective_dict.values()]
    else:
        normalized_values = [(val - min_obj) / (max_obj - min_obj) for val in objective_dict.values()]

    # Create a mapping from schedule to normalized objective value
    schedule_to_obj = {schedule: obj for schedule, obj in zip(G.nodes(), normalized_values)}

    # 7. Create edge traces (for visual reference, lines without direction)
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    # 8. Create node traces with updated hover info
    node_x = []
    node_y = []
    node_labels = []      # For node annotations (schedule)
    node_hovertext = []   # For hover info (objective value)
    node_colors = []

    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        node_x.append(x)
        node_y.append(y)
        # Node label shows the schedule
        node_labels.append(str(node))
        # Hover text shows the objective value
        node_hovertext.append(f"Objective: {objective_dict[node]:.2f}")
        node_colors.append(schedule_to_obj[node])

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_labels,
        textposition="top center",
        hovertext=node_hovertext,
        hoverinfo='text',
        marker=dict(
            showscale=True,
            # Define colorscale from deep red to white
            colorscale=[
                [0.0, 'rgb(165,0,38)'],    # Deep Red
                [1.0, 'rgb(255,255,255)']  # White
            ],
            reversescale=True,  # Set to True as per your changes
            color=node_colors,
            size=10,            # Marker size
            colorbar=dict(
                thickness=15,
                title=dict(
                    text='Objective Value',
                    side='right'
                ),
                xanchor='left',
            ),
            line_width=2
        )
    )

    # 9. Prepare arrow annotations based on directed edges with edge labels
    arrow_annotations = []
    for edge in G.edges():
        source, target = edge
        # Compare objective values to determine direction
        source_obj = objective_dict[source]
        target_obj = objective_dict[target]
        if source_obj > target_obj:
            # Arrow from source to target with negative difference
            diff = target_obj - source_obj  # Negative value
            x0, y0 = G.nodes[source]['pos']
            x1, y1 = G.nodes[target]['pos']
            # Calculate midpoint for label
            mid_x = (x0 + x1) / 2
            mid_y = (y0 + y1) / 2
            # Add arrow annotation
            arrow_annotations.append(dict(
                x=x1,
                y=y1,
                ax=x0,
                ay=y0,
                xref='x',
                yref='y',
                axref='x',
                ayref='y',
                showarrow=True,
                arrowhead=2,
                arrowsize=2.5,    # Adjusted arrow size
                arrowwidth=1,
                arrowcolor="#000000",
                opacity=0.5
            ))
            # Add text annotation for the difference
            arrow_annotations.append(dict(
                x=mid_x,
                y=mid_y,
                text=f"{diff:.2f}",
                showarrow=False,
                xref='x',
                yref='y',
                font=dict(color='blue', size=10),
                align='center'
            ))
        elif target_obj > source_obj:
            # Arrow from target to source with negative difference
            diff = source_obj - target_obj  # Negative value
            x0, y0 = G.nodes[target]['pos']
            x1, y1 = G.nodes[source]['pos']
            # Calculate midpoint for label
            mid_x = (x0 + x1) / 2
            mid_y = (y0 + y1) / 2
            # Add arrow annotation
            arrow_annotations.append(dict(
                x=x1,
                y=y1,
                ax=x0,
                ay=y0,
                xref='x',
                yref='y',
                axref='x',
                ayref='y',
                showarrow=True,
                arrowhead=2,
                arrowsize=2.5,    # Adjusted arrow size
                arrowwidth=1,
                arrowcolor="#000000",
                opacity=0.5
            ))
            # Add text annotation for the difference
            arrow_annotations.append(dict(
                x=mid_x,
                y=mid_y,
                text=f"{diff:.2f}",
                showarrow=False,
                xref='x',
                yref='y',
                font=dict(color='blue', size=10),
                align='center'
            ))
        # If equal, no arrow or label

    # 10. Assemble the figure with arrow annotations
    exp_s = sum([i * si for i, si in enumerate(s)])  # Calculate expected service time
    fig = go.Figure(data=[edge_trace, node_trace],
                 layout=go.Layout(
                    title=dict(
                        text=f"Network of Schedules with N={N} Patients, T={T} Time Intervals, Exp. service time = {exp_s:.2f}, and w = {w}",
                        font=dict(size=16)
                    ),
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    annotations=arrow_annotations + [ dict(
                        text="Created with NetworkX and Plotly",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002 ) ],
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )

    return fig

def create_schedule_network_from_lists(
    schedules: List[List[int]],
    objective_values: List[float],
    echo: bool = False
) -> go.Figure:
    """
    Creates a network graph connecting each schedule to its subsequent schedule.
    Edge lengths are proportional to the difference in objective values between connected schedules.
    Each node is offset vertically to prevent overlap.
    The color scale ranges from deep red (max objective) to white (min objective).
    The difference value is annotated next to each edge.
    
    Parameters:
    - schedules (List[List[int]]): A list of schedules, each schedule is a list of patient allocations across time intervals.
    - objective_values (List[float]): A list of objective values corresponding to each schedule.
    - echo (bool): If True, prints schedule details and objective values.
    
    Returns:
    - fig (plotly.graph_objects.Figure): Interactive network graph.
    """
    
    # 1. Validate Inputs
    if len(schedules) != len(objective_values):
        raise ValueError("The number of schedules must match the number of objective values.")
    
    num_schedules = len(schedules)
    
    # 2. Initialize the directed graph
    G = nx.DiGraph()
    
    # 3. Add nodes with objective values
    for idx, (schedule, obj) in enumerate(zip(schedules, objective_values)):
        G.add_node(idx, schedule=schedule, objective=obj)
        if echo:
            print(f"Schedule {idx}: {schedule}, Objective: {obj:.2f}")
    
    # 4. Add edges connecting each schedule to the next
    for idx in range(num_schedules - 1):
        G.add_edge(idx, idx + 1)
    
    # 5. Assign positions to nodes based on cumulative objective differences
    x_positions = [0]
    y_positions = [0]  # Initialize y_positions with the first node at y=0
    vertical_offset = 5.5  # Adjust this value to control vertical spacing
    
    for i in range(1, num_schedules):
        diff = abs(objective_values[i] - objective_values[i-1])
        x = x_positions[-1] + diff
        # Alternate the vertical position to offset nodes
        y = y_positions[-1] - vertical_offset
        x_positions.append(x)
        y_positions.append(y)
    
    pos = {idx: (x, y) for idx, (x, y) in enumerate(zip(x_positions, y_positions))}
    
    # 6. Create edge traces
    edge_traces = []
    annotations = []
    
    for edge in G.edges():
        source, target = edge
        x0, y0 = pos[source]
        x1, y1 = pos[target]
        
        # Edge line
        edge_traces.append(
            go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=2, color='gray'),
                hoverinfo='none'
            )
        )
        
        # Calculate midpoint for annotation
        mid_x = (x0 + x1) / 2
        mid_y = (y0 + y1) / 2
        
        # Objective difference
        obj_diff = objective_values[target] - objective_values[source]
        
        # Edge annotation
        annotations.append(
            dict(
                x=mid_x,
                y=mid_y,
                text=f"{obj_diff:.2f}",
                showarrow=True,
                arrowhead=2,
                ax=mid_x,
                ay=mid_y + (0.3 if y1 >= y0 else -0.3),  # Offset direction based on edge slope
                xref='x',
                yref='y',
                font=dict(color='blue', size=12),
                align='center'
            )
        )
    
    # 7. Create node trace
    node_trace = go.Scatter(
        x=x_positions,
        y=y_positions,
        mode='markers+text',
        text=[str(schedule) for schedule in schedules],
        textposition="top center",
        marker=dict(
            size=15,
            color=objective_values,
            colorscale=[
                [0.0, 'rgb(255,255,255)'],    # White for min objective
                [1.0, 'rgb(165,0,38)']        # Deep Red for max objective
            ],
            cmin=min(objective_values),
            cmax=max(objective_values),
            colorbar=dict(
                title='Objective Value',
                titleside='right',
                tickmode='linear',
                ticks='outside'
            ),
            reversescale=False,  # False since our colorscale is defined from min to max
            line=dict(width=2, color='black')
        ),
        hoverinfo='text'
    )
    
    # Update hover text to include schedule and objective value
    node_trace.text = [f"Schedule: {schedule}<br>Objective: {obj:.2f}" for schedule, obj in zip(schedules, objective_values)]
    
    # 8. Assemble the figure
    fig = go.Figure(
        data=edge_traces + [node_trace],
        layout=go.Layout(
            title=dict(
                text=f"Local search trace for schedule with N={sum(schedules[0])} patients",
                x=0.5,
                xanchor='center',
                font=dict(size=20)
            ),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=50, l=50, r=150, t=100),
            annotations=annotations,
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                title='Objective Value Difference'
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False
            ),
            template="plotly_white"
        )
    )
    
    return fig


