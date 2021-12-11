import numpy as np
import time
from utils import recurrence_H, divides, clash_matrix, payoff_matrix_pandas, \
    payoff_matrix_pandas_multithreading, blotto, majoritarian, chopstick, attack

#%%

def test_H(A, n):
    strats = divides(A,n)
    errors = 0
    tries = 0
    for i in range(strats.shape[0]):
        for j in range(strats.shape[0]):
            tries += 1
            mock_A = strats[i]
            mock_B = strats[j]
            if(not np.all(recurrence_H(mock_A, mock_B)[-1,-1,:,:] == h(mock_A, mock_B))):
                errors += 1
                print("===================")
                print(mock_A, mock_B)
                print(clash_matrix(mock_A, mock_B))
    print("Number of errors", errors, "on", tries, "tries")
# test_H(10, 4)

def test_payoff_matrix_time(A, B, n, aggregation):
    start = time.time()
    matrix = payoff_matrix_pandas(A, B, n, aggregation)
    print(time.time() - start)

def test_payoff_matrix_time_multithreading(A, B, n, aggregation, threads_number=None):
    start = time.time()
    matrix = payoff_matrix_pandas_multithreading(A, B, n, aggregation, threads_number)
    print(time.time() - start)

test_payoff_matrix_time(10,10,5,chopstick)
test_payoff_matrix_time_multithreading(10,10,5,chopstick,1)
test_payoff_matrix_time_multithreading(10,10,5,chopstick)
