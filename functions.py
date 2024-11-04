import math
import random
import json
import numpy as np
import time
import pickle
from itertools import combinations_with_replacement, combinations
from typing import List, Generator, Tuple

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
    u = np.zeros(T)
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
