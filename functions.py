import math
import random
import numpy as np
from itertools import combinations_with_replacement
from typing import List, Generator

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

def create_neighbors_list(S: list[list[int]]) -> list[(list[int], list[int])]: # Create a set of pairs of schedules that are from the same neighborhood
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
    s = service_time_with_no_shows(s, q) # Adjust service times distribution for no-shows
    sp = np.array([1], dtype=np.int64) # Set probability of first spillover time being zero to 1
    wt_list = [] # Initialize wt_list for saving all waiting times for all patients in the schedule
    ewt = 0 # Initialize sum of expected waiting times
    for x in schedule: # For each interval -
      if(x == 0): # In case there are no patients,
        wt_temp = [np.array(sp)] # the spillover from the previous interval is recorded,
        wt_list.append([]) # but there are no waiting times.
        sp = [] # Initialize the spillover time distribution 
        sp.append(np.sum(wt_temp[-1][:d+1])) # All the work from the previous interval's spillover that could not be processed will be added to the this interval's spillover.
        sp[1:] = wt_temp[-1][d+1:]
      else: # In case there are patients scheduled,
        wt_temp = [np.array(sp)] # Initialize wt_temp for saving all waiting times for all patients in the interval. The first patient has to wait for the spillover work from the previous period.
        ewt += np.dot(range(len(sp)), sp) # Add waiting time for first patient in interval
        for i in range(x-1): # For each patient
          wt = np.convolve(wt_temp[i], s) # Calculate the waiting time distribution
          wt_temp.append(wt)
          ewt += np.dot(range(len(wt)), wt)
        wt_list.append(wt_temp)
        sp = []
        sp.append(np.sum(np.convolve(wt_temp[-1],s)[:d+1])) # Calculate the spillover
        sp[1:] = np.convolve(wt_temp[-1],s)[d+1:]
    return ewt
  
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
