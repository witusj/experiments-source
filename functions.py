import math
import random
import json
import numpy as np
from itertools import combinations_with_replacement, combinations
from typing import List, Generator

def get_v_star(T):
    # Create an initial vector 'u' of zeros with length 'T'
    u = np.zeros(T)
    # Set the first element of vector 'u' to -1 and the last element to 1
    u[0] = -1
    u[-1] = 1
    # Initialize the list 'v_star' with the initial vector 'u'
    v_star = [u.copy()]
    
    # Loop over the length of 'u' minus one times to generate shifted versions of 'u'
    for i in range(T - 1):
        # Rotate 'u' by moving the last element to the front
        u = np.roll(u, 1)
        # Append the updated vector 'u' to the list 'v_star'
        v_star.append(u.copy())
    
    # Return 'v_star' as a list of lists, which is easier to process in the main function
    return np.array(v_star)

def generate_all_schedules(N: int, T: int) -> list[list[int]]:
    def generate(current_schedule: list[int], remaining_patients: int, remaining_slots: int):
        if remaining_slots == 0:
            if remaining_patients == 0:
                schedules.append(current_schedule)
            return
        
        for i in range(remaining_patients + 1):
            generate(current_schedule + [i], remaining_patients - i, remaining_slots - 1)
    
    pop_size = math.comb(N+T-1, N)
    print(f"Number of possible schedules: {pop_size}")
    schedules = []
    generate([], N, T)
    return schedules

def serialize_schedules(N: int, T: int) -> None:
    file_path=f"experiments/n{N}t{T}.pickle"
    
    # Start timing
    start_time = time.time()
    
    # Open a file for writing serialized schedules
    with open(file_path, 'wb') as f:
        for comb in combinations_with_replacement(range(T), N):
            schedule = np.bincount(comb, minlength=T).tolist()
            pickle.dump(schedule, f)
    
    # End timing
    end_time = time.time()
    
    print(f"Number of possible schedules: {math.comb(N+T-1, N)}")
    print(f"Time taken: {end_time - start_time} seconds")
    print(f"Schedules are serialized and saved to {file_path}")

def load_schedules(file_path: [str]) -> list[list[int]]:
    # Start timing
    start_time = time.time()
    
    schedules = []
    with open(file_path, 'rb') as f:
        while True:
            try:
                schedules.append(pickle.load(f))
            except EOFError:
                break
        # End timing
    end_time = time.time()
    print(f"Loading time: {end_time - start_time} seconds")
    return schedules

def generate_random_schedule(N: int, T: int) -> list[int]:
  """
  A function to generate a random schedule for N patients and T slots
  """
  slots = random.choices(range(T), k = N)
  schedule = [0]*T
  for slot in slots:
    schedule[slot] += 1
  return(schedule)

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
    print(f"Total number of combinations: {total_combinations}")

    schedules = []
    for _ in range(num_samples):
        # Randomly select an index
        random_index = random.randint(0, total_combinations - 1)
        # Convert index to combination
        sample = index_to_combination(random_index, T, N)
        schedule = np.bincount(sample, minlength=T).tolist()
        schedules.append(schedule)

    return schedules
  
def create_neighbors_list(s: list[int]) -> (list[int], list[int]):
    """
    Create a set of pairs of schedules that are from the same neighborhood.
    
    Parameters:
      s (list[int]): A list of integers with |s| = T and sum N.
      
    Returns:
      tuple(list[int], list[int]): A pair of schedules.
    """
    # Create a set of vectors of length T
    v_star = get_v_star(len(s))
    
    # Choose a random element of t with probability P(t = i) = C(T,i)/((2^T)-2) for i in [1, ..., T-1]
    i = random.choices(range(1, len(s)), weights=[math.comb(len(s), i) for i in range(1, len(s))])[0]
    
    # Create a list l of all subsets of t with length i
    l = list(combinations(range(len(s)), i))
    
    # Choose a random element of l with probability 1/|l| and save it as j
    j = random.choice(l)
    
    # Select all elements of V* with index in j and save them as V_j
    # Sum the corresponding vectors in v_star
    v_j = [v_star[idx] for idx in j]  # Convert NumPy arrays to lists
    
    # Sum the elements of v_j and s and save as s_p
    s_p = s.copy()
    for v in v_j:
      s_p_temp = [int(x + y) for x, y in zip(s_p, v)]
      if np.all(np.array(s_p_temp) >= 0):
        s_p = s_p_temp
        
    return s, s_p
    
def create_neighbors_list_single_swap(S: list[list[int]]) -> list[(list[int], list[int])]: # Create a set of pairs of schedules that are from the same neighborhood
    neighbors_list = []
    
    # For each schedule in in the subset choose 2 random intervals i, j and swap 1 patient
    for s in S:
      i = random.choice(range(len(s)))  # Ensure i is a valid index in s
      J = [index for index, element in enumerate(s) if element > 0 and index != i]
      
      if not J:  # Ensure j is not empty
          continue
      
      j = random.choice(J)  # Choose a random valid index from j
      
      s_pair = s.copy()  # Create a copy of s to modify
      s_pair[i] = s[i] + 1
      s_pair[j] = s[j] - 1
      
      neighbors_list.append((s, s_pair))
    
    return(neighbors_list)

def calculate_objective(schedule: list[int], s: list[float], d: int, q: float) -> float:
    """
    Calculate the objective value based on the given schedule and parameters.

    This function adjusts the service times distribution for no-shows, calculates 
    the waiting times for all patients in the schedule, sums the expected 
    waiting times, and calculate the spillover time for the last interval (= overtime).

    Parameters:
    - schedule (list[int]): A list representing the schedule intervals.
    - s (list[float]): Service times distribution.
    - d (int): Some integer parameter (likely related to time or patients).
    - q (float): Probability related to no-shows.

    Returns:
    - tuple: 
        - float: The sum of expected waiting times.
        - float: The expected spillover time.
    """
    s = service_time_with_no_shows(s, q)
    sp = np.array([1], dtype=np.int64)
    wt_list = []
    ewt = 0
    for x in schedule:
        if x == 0:
            wt_temp = [np.array(sp)]
            wt_list.append([])
            sp = []
            sp.append(np.sum(wt_temp[-1][:d+1]))
            sp[1:] = wt_temp[-1][d+1:]
        else:
            wt_temp = [np.array(sp)]
            ewt += np.dot(range(len(sp)), sp)
            for i in range(x-1):
                wt = np.convolve(wt_temp[i], s)
                wt_temp.append(wt)
                ewt += np.dot(range(len(wt)), wt)
            wt_list.append(wt_temp)
            sp = []
            sp.append(np.sum(np.convolve(wt_temp[-1], s)[:d+1]))
            sp[1:] = np.convolve(wt_temp[-1], s)[d+1:]
        esp = np.dot(range(len(sp)), sp)  
    return ewt, esp
  
def service_time_with_no_shows(s, q):
    # """
    # Function to adjust a distribution of service times for no-shows
    # 
    # Args:
    #     s (numpy.ndarray): An array with service times.
    #     q (double): The fraction of no-shows.
    # 
    # Returns:
    #     numpy.ndarray: The adjusted array of service times.
    # """
    
    s_adj = [(1 - q) * float(si) for si in s]
    s_adj[0] = s_adj[0] + q
    return(s_adj)

def calculate_ambiguousness(y_pred_proba: np.ndarray) -> np.ndarray:
    """
    Calculate the ambiguousness for each array of probabilities.

    Parameters:
    y_pred_proba (np.ndarray): Array of shape (n_samples, n_classes) with predicted probabilities for each class.

    Returns:
    np.ndarray: Array of ambiguousness for each sample.
    """
    # Ensure probabilities are in numpy array
    y_pred_proba = np.array(y_pred_proba)
    
    # Define a small bias term
    epsilon = 1e-10
    
    # Add small bias term to probabilities that contain [0, 1] or [1, 0]
    for i in range(len(y_pred_proba)):
        if np.any(y_pred_proba[i] == 0):
            print(f"{y_pred_proba[i]}: There is at least one zero in the array. Added bias term {epsilon}.")
            # Add bias term
            y_pred_proba[i] = np.clip(y_pred_proba[i], epsilon, 1 - epsilon)
    
    # Calculate ambiguousness for each array of probabilities
    ambiguousness = -np.sum(y_pred_proba * np.log2(y_pred_proba), axis=1)
    
    return ambiguousness

def calculate_opaqueness(y_pred_proba: np.ndarray) -> np.ndarray:
    """
    Calculate the opaqueness for each array of normalized predicted objectives.

    Parameters:
    y_pred_proba (np.ndarray): Array of normalized predicted objectives.

    Returns:
    np.ndarray: Array of ambiguousness for each sample.
    """
    # Ensure probabilities are in numpy array
    y_pred_proba = np.array(y_pred_proba)
    
    # Define a small bias term
    epsilon = 1e-10
    
    # Add small bias term to probabilities that contain [0, 1] or [1, 0]
    for i in range(len(y_pred_proba)):
        if np.any(y_pred_proba[i] == 0):
            print(f"{y_pred_proba[i]}: There is at least one zero in the array. Added bias term {epsilon}.")
            # Add bias term
            y_pred_proba[i] = np.clip(y_pred_proba[i], epsilon, 1 - epsilon)
    
    # Calculate ambiguousness for each array of probabilities
    opaqueness = -np.sum(y_pred_proba * np.log2(y_pred_proba))
    
    return opaqueness

def compare_json(json1, json2):
    """
    Compares two JSON objects and returns a dictionary with the differences.
    
    Parameters:
        json1 (dict): The first JSON object to compare.
        json2 (dict): The second JSON object to compare.
    
    Returns:
        dict: A dictionary showing the differences between the two JSON objects.
    """
    differences = {}
    
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
    
    for key in json2.keys():
        if key not in json1:
            differences[key] = {
                "json1_value": "Key not found in json1",
                "json2_value": json2[key]
            }

    return differences
