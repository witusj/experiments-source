import unittest
import numpy as np
from typing import List, Tuple, Dict
import copy
import time
import matplotlib.pyplot as plt
from functools import partial

from functions_new import service_time_with_no_shows, compute_convolutions, compute_convolutions_fft, calculate_objective_serv_time_lookup

import numpy as np
import time

def fft_test():
    # Generate a random PMF of size 1000
    probabilities = np.random.rand(1000)
    probabilities /= np.sum(probabilities)
    N = 20
    q = 0.1
    iterations = 10

    # Time the standard (direct) convolution method
    start = time.time()
    for _ in range(iterations):
        conv_std = compute_convolutions(probabilities, N, q)
    std_time = time.time() - start

    # Time the FFT-based convolution method
    start = time.time()
    for _ in range(iterations):
        conv_fft = compute_convolutions_fft(probabilities, N, q)
    fft_time = time.time() - start

    print(f"Standard convolution time: {std_time:.6f} seconds")
    print(f"FFT convolution time: {fft_time:.6f} seconds")
    
    #  Test the results are the same
    for k in range(1, N + 1):
        np.testing.assert_array_almost_equal(conv_std[k], conv_fft[k])

class TestServiceTimeNoShows(unittest.TestCase):
    """Test class for service_time_with_no_shows function."""
    
    def test_no_show_adjustment(self):
        """Test that the service time is correctly adjusted for no-shows."""
        service_time = [0.1, 0.2, 0.3, 0.4]
        q = 0.2  # 20% no-show probability
        
        adjusted = service_time_with_no_shows(service_time, q)
        
        # Check first element (should include no-show probability)
        self.assertAlmostEqual(adjusted[0], 0.1 * 0.8 + 0.2)
        
        # Check other elements (should be scaled by (1-q))
        for i in range(1, len(service_time)):
            self.assertAlmostEqual(adjusted[i], service_time[i] * 0.8)
    
    def test_zero_no_show_probability(self):
        """Test with zero no-show probability."""
        service_time = [0.1, 0.2, 0.3, 0.4]
        q = 0.0
        
        adjusted = service_time_with_no_shows(service_time, q)
        
        # Should be identical to original
        for i in range(len(service_time)):
            self.assertAlmostEqual(adjusted[i], service_time[i])
    
    def test_full_no_show_probability(self):
        """Test with 100% no-show probability."""
        service_time = [0.1, 0.2, 0.3, 0.4]
        q = 1.0
        
        adjusted = service_time_with_no_shows(service_time, q)
        
        # First element should be 1.0, others should be 0
        self.assertAlmostEqual(adjusted[0], 1.0)
        for i in range(1, len(service_time)):
            self.assertAlmostEqual(adjusted[i], 0.0)
    
    def test_probability_sum(self):
        """Test that the sum of probabilities remains 1."""
        service_time = [0.1, 0.2, 0.3, 0.4]
        q_values = [0.0, 0.2, 0.5, 0.8, 1.0]
        
        for q in q_values:
            adjusted = service_time_with_no_shows(service_time, q)
            self.assertAlmostEqual(sum(adjusted), 1.0)


class TestComputeConvolutions(unittest.TestCase):
    """Test class for compute_convolutions function."""
    
    def setUp(self):
        """Set up test cases."""
        # Simple PMF
        self.simple_pmf = [0.5, 0.5]
        
        # Service time distribution with mean 5
        self.service_time = np.zeros(11)
        self.service_time[3] = 0.2
        self.service_time[5] = 0.6
        self.service_time[8] = 0.2
    
    def test_convolution_keys(self):
        """Test that the convolutions dictionary has the correct keys."""
        N = 10  # Total number of patients across all slots
        convolutions = compute_convolutions(self.simple_pmf, N)
        
        # Should have keys from 1 to N
        for k in range(1, N + 1):
            self.assertIn(k, convolutions)
    
    def test_first_convolution(self):
        """Test that the first convolution is the adjusted service time."""
        q = 0.2
        convolutions = compute_convolutions(self.simple_pmf, 10, q)
        
        expected = service_time_with_no_shows(self.simple_pmf, q)
        np.testing.assert_array_almost_equal(convolutions[1], expected)
    
    def test_second_convolution(self):
        """Test that the second convolution is computed correctly."""
        convolutions = compute_convolutions(self.simple_pmf, 10)
        
        # For simple_pmf = [0.5, 0.5], the 2-fold convolution should be [0.25, 0.5, 0.25]
        expected = np.array([0.25, 0.5, 0.25])
        np.testing.assert_array_almost_equal(convolutions[2], expected)
    
    def test_preserves_probability(self):
        """Test that all convolutions preserve total probability of 1."""
        N = 10
        convolutions = compute_convolutions(self.service_time, N)
        
        for k in range(1, N + 1):
            self.assertAlmostEqual(sum(convolutions[k]), 1.0)


class TestCalculateObjectiveServTimeLookup(unittest.TestCase):
    """Test class for calculate_objective_serv_time_lookup function."""
    
    def setUp(self):
        """Set up test cases."""
        self.d = 5  # Duration threshold
        self.q = 0.2  # No-show probability
        
        # Create simple service time distribution
        self.service_time = np.zeros(11)
        self.service_time[3] = 0.2
        self.service_time[5] = 0.6
        self.service_time[8] = 0.2
    
    def compute_convolutions_for_schedule(self, schedule):
        """Helper method to compute convolutions for a specific schedule."""
        total_patients = sum(schedule)  # Sum of all patients in the schedule
        if total_patients == 0:
            total_patients = 1  # Ensure at least one convolution for empty schedules
        
        return compute_convolutions(self.service_time, total_patients, self.q)
    
    def test_empty_schedule(self):
        """Test with an empty schedule."""
        schedule = []
        convolutions = self.compute_convolutions_for_schedule(schedule)
        
        ewt, esp = calculate_objective_serv_time_lookup(schedule, self.d, self.q, convolutions)
        
        # Empty schedule should result in zero waiting time and zero spillover
        self.assertEqual(ewt, 0)
        self.assertEqual(esp, 0)
    
    def test_single_patient(self):
        """Test with a single patient."""
        schedule = [1]
        convolutions = self.compute_convolutions_for_schedule(schedule)
        
        ewt, esp = calculate_objective_serv_time_lookup(schedule, self.d, self.q, convolutions)
        
        # Single patient should have zero waiting time
        # Spillover depends on service time distribution and duration threshold
        self.assertEqual(ewt, 0)
        self.assertGreaterEqual(esp, 0)
    
    def test_multiple_patients_same_slot(self):
        """Test with multiple patients in the same slot."""
        schedule = [3]
        convolutions = self.compute_convolutions_for_schedule(schedule)
        
        ewt, esp = calculate_objective_serv_time_lookup(schedule, self.d, self.q, convolutions)
        
        # Should have positive waiting time and spillover
        self.assertGreater(ewt, 0)
        self.assertGreaterEqual(esp, 0)
    
    def test_multiple_slots(self):
        """Test with multiple slots."""
        schedule = [2, 0, 3, 1]
        convolutions = self.compute_convolutions_for_schedule(schedule)
        
        ewt, esp = calculate_objective_serv_time_lookup(schedule, self.d, self.q, convolutions)
        
        # Should have positive waiting time and spillover
        self.assertGreaterEqual(ewt, 0)
        self.assertGreaterEqual(esp, 0)
    
    def test_zero_patients(self):
        """Test with zero patients in some slots."""
        schedule1 = [2, 2]
        convolutions1 = self.compute_convolutions_for_schedule(schedule1)
        
        schedule2 = [2, 0, 2]
        convolutions2 = self.compute_convolutions_for_schedule(schedule2)
        
        ewt1, esp1 = calculate_objective_serv_time_lookup(schedule1, self.d, self.q, convolutions1)
        ewt2, esp2 = calculate_objective_serv_time_lookup(schedule2, self.d, self.q, convolutions2)
        
        # Schedule with a break should have less or equal waiting time
        self.assertLessEqual(ewt2, ewt1)
    
    def test_deterministic_results(self):
        """Test that results are deterministic for the same inputs."""
        schedule = [2, 1, 3]
        convolutions = self.compute_convolutions_for_schedule(schedule)
        
        ewt1, esp1 = calculate_objective_serv_time_lookup(schedule, self.d, self.q, convolutions)
        ewt2, esp2 = calculate_objective_serv_time_lookup(schedule, self.d, self.q, convolutions)
        
        self.assertEqual(ewt1, ewt2)
        self.assertEqual(esp1, esp2)
    
    def test_various_schedules(self):
        """Test with different schedule patterns."""
        test_schedules = [
            [1, 1, 1, 1],         # Uniform distribution
            [4, 0, 0, 0],         # Front-loaded
            [0, 0, 0, 4],         # Back-loaded
            [2, 0, 2, 0],         # Alternating
            [1, 2, 3, 4],         # Increasing
            [4, 3, 2, 1]          # Decreasing
        ]
        
        # Verify all schedules work correctly
        results = []
        for schedule in test_schedules:
            convolutions = self.compute_convolutions_for_schedule(schedule)
            ewt, esp = calculate_objective_serv_time_lookup(schedule, self.d, self.q, convolutions)
            
            # Store results for comparison
            results.append((schedule, ewt, esp))
            
            # Basic validity checks
            self.assertGreaterEqual(ewt, 0)
            self.assertGreaterEqual(esp, 0)
        
        # Compare front-loaded vs back-loaded schedules
        front_loaded = results[1]
        back_loaded = results[2]
        
        print(f"\nFront-loaded schedule {front_loaded[0]}: EWT = {front_loaded[1]}, ESP = {front_loaded[2]}")
        print(f"Back-loaded schedule {back_loaded[0]}: EWT = {back_loaded[1]}, ESP = {back_loaded[2]}")
        
        # Compare increasing vs decreasing schedules
        increasing = results[4]
        decreasing = results[5]
        
        print(f"Increasing schedule {increasing[0]}: EWT = {increasing[1]}, ESP = {increasing[2]}")
        print(f"Decreasing schedule {decreasing[0]}: EWT = {decreasing[1]}, ESP = {decreasing[2]}")


class SpeedTests(unittest.TestCase):
    """Performance tests for the scheduling functions."""
    
    def setUp(self):
        """Set up test data for speed tests."""
        # Parameters
        self.d = 10
        self.q = 0.2
        
        # Create service time distribution
        self.service_time = np.zeros(21)
        self.service_time[5] = 0.2
        self.service_time[10] = 0.6
        self.service_time[15] = 0.2
        
        # Different schedule sizes for testing
        self.small_schedule = [2, 3, 0, 1]  # 6 patients
        self.medium_schedule = [2, 3, 0, 1, 4, 2, 0, 3, 1, 2]  # 18 patients
        self.large_schedule = [2, 3, 0, 1, 4, 2, 0, 3, 1, 2] * 5  # 90 patients
        
        # Precompute convolutions for each schedule
        self.small_convolutions = compute_convolutions(self.service_time, 6, self.q)
        self.medium_convolutions = compute_convolutions(self.service_time, 18, self.q)
        self.large_convolutions = compute_convolutions(self.service_time, 90, self.q)
    
    def test_speed_service_time_with_no_shows(self):
        """Test the speed of service_time_with_no_shows function."""
        iterations = 1000
        start_time = time.time()
        
        for _ in range(iterations):
            service_time_with_no_shows(self.service_time, self.q)
        
        elapsed = time.time() - start_time
        print(f"\nTime for {iterations} calls to service_time_with_no_shows: {elapsed:.4f} seconds")
        print(f"Average time per call: {elapsed / iterations * 1000:.4f} ms")
    
    def test_speed_compute_convolutions(self):
        """Test the speed of compute_convolutions function with different N values."""
        iterations = 10
        n_values = [6, 18, 50, 90]  # Corresponding to total patients in different schedules
        
        results = []
        for n in n_values:
            start_time = time.time()
            
            for _ in range(iterations):
                compute_convolutions(self.service_time, n, self.q)
            
            elapsed = time.time() - start_time
            results.append((n, elapsed / iterations))
            print(f"\nTime for {iterations} calls to compute_convolutions with N={n}: {elapsed:.4f} seconds")
            print(f"Average time per call: {elapsed / iterations * 1000:.4f} ms")
        
        # Plot the results
        plt.figure(figsize=(10, 6))
        plt.plot([x[0] for x in results], [x[1] * 1000 for x in results], marker='o')
        plt.xlabel('N (total patients in schedule)')
        plt.ylabel('Average time (ms)')
        plt.title('Performance of compute_convolutions with different N values')
        plt.grid(True)
        plt.savefig('convolution_speed_test.png')
    
    def test_speed_calculate_objective(self):
        """Test the speed of calculate_objective_serv_time_lookup with different schedule sizes."""
        iterations = 100
        
        schedules_with_convolutions = [
            ("Small (6 patients)", self.small_schedule, self.small_convolutions),
            ("Medium (18 patients)", self.medium_schedule, self.medium_convolutions),
            ("Large (90 patients)", self.large_schedule, self.large_convolutions)
        ]
        
        results = []
        for name, schedule, convolutions in schedules_with_convolutions:
            start_time = time.time()
            
            for _ in range(iterations):
                calculate_objective_serv_time_lookup(schedule, self.d, self.q, convolutions)
            
            elapsed = time.time() - start_time
            results.append((name, elapsed / iterations))
            print(f"\nTime for {iterations} calls to calculate_objective with {name}: {elapsed:.4f} seconds")
            print(f"Average time per call: {elapsed / iterations * 1000:.4f} ms")
        
        # Plot the results
        plt.figure(figsize=(10, 6))
        plt.bar([x[0] for x in results], [x[1] * 1000 for x in results])
        plt.ylabel('Average time (ms)')
        plt.title('Performance of calculate_objective_serv_time_lookup with different schedule sizes')
        plt.grid(True, axis='y')
        plt.savefig('objective_speed_test.png')
    
    def test_end_to_end_performance(self):
        """Test the end-to-end performance (compute convolutions + calculate objective)."""
        iterations = 10
        
        schedules = [
            ("Small (6 patients)", self.small_schedule),
            ("Medium (18 patients)", self.medium_schedule),
            ("Large (90 patients)", self.large_schedule)
        ]
        
        results = []
        for name, schedule in schedules:
            start_time = time.time()
            
            for _ in range(iterations):
                # Compute convolutions for this schedule
                convolutions = compute_convolutions(self.service_time, sum(schedule), self.q)
                
                # Calculate objective using these convolutions
                calculate_objective_serv_time_lookup(schedule, self.d, self.q, convolutions)
            
            elapsed = time.time() - start_time
            results.append((name, elapsed / iterations))
            print(f"\nTime for {iterations} end-to-end calculations with {name}: {elapsed:.4f} seconds")
            print(f"Average time per calculation: {elapsed / iterations * 1000:.4f} ms")
        
        # Plot the results
        plt.figure(figsize=(10, 6))
        plt.bar([x[0] for x in results], [x[1] * 1000 for x in results])
        plt.ylabel('Average time (ms)')
        plt.title('End-to-End Performance (convolutions + objective calculation)')
        plt.grid(True, axis='y')
        plt.savefig('end_to_end_performance.png')


def test_schedule_variations():
    """Test how different schedule configurations affect waiting time and spillover."""
    # Parameters
    d = 5
    q = 0.2
    
    # Create service time distribution
    service_time = np.zeros(11)
    service_time[3] = 0.2
    service_time[5] = 0.6
    service_time[8] = 0.2
    
    # Different schedule patterns with the same total number of patients (8)
    schedules = [
        ("Uniform", [2, 2, 2, 2]),
        ("Decreasing", [5, 2, 1, 0]),
        ("Increasing", [0, 1, 2, 5]),
        ("Front-heavy", [4, 4, 0, 0]),
        ("Back-heavy", [0, 0, 4, 4]),
        ("Alternating", [4, 0, 4, 0]),
        ("Bailey-rule", [2, 1, 1, 1, 1, 1, 1])  # Schedule 2 initially, then 1 each slot
    ]
    
    # Compute convolutions once (need convolutions up to 8)
    convolutions = compute_convolutions(service_time, 8, q)
    
    results = []
    print("\n=== Schedule Variation Test Results ===")
    
    for name, schedule in schedules:
        # Calculate metrics
        ewt, esp = calculate_objective_serv_time_lookup(schedule, d, q, convolutions)
        
        # Store results
        results.append((name, schedule, ewt, esp, ewt + esp))
        
        print(f"{name} schedule {schedule}:")
        print(f"  Expected Waiting Time: {ewt:.4f}")
        print(f"  Expected Spillover: {esp:.4f}")
        print(f"  Total (EWT + ESP): {ewt + esp:.4f}")
        print("")
    
    # Sort by total objective (EWT + ESP)
    sorted_results = sorted(results, key=lambda x: x[4])
    
    print("Schedules ranked by total objective (EWT + ESP):")
    for i, (name, schedule, ewt, esp, total) in enumerate(sorted_results, 1):
        print(f"{i}. {name} {schedule}: {total:.4f}")
    
    # Create bar chart of results
    plt.figure(figsize=(12, 8))
    
    # Extract data for plotting
    names = [r[0] for r in results]
    ewt_values = [r[2] for r in results]
    esp_values = [r[3] for r in results]
    
    # Set up bar chart
    x = np.arange(len(names))
    width = 0.35
    
    plt.bar(x - width/2, ewt_values, width, label='Expected Waiting Time')
    plt.bar(x + width/2, esp_values, width, label='Expected Spillover')
    
    plt.xlabel('Schedule Configuration')
    plt.ylabel('Time')
    plt.title('Impact of Different Schedule Configurations on Waiting Time and Spillover')
    plt.xticks(x, names, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.grid(True, axis='y')
    plt.savefig('schedule_comparison.png')


if __name__ == '__main__':
    # Run the comparison test between FFT and direct convolution
    fft_test()
    
    # Run the unit tests
    unittest.main(exit=False)
    
    # Run additional tests
    test_schedule_variations()
