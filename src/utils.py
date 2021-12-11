import numpy as np
import pandas as pd
from itertools import permutations
import math
import scipy.special
import concurrent.futures

def next_divide(divide):
    n = divide.shape[1]
    div_num = divide.shape[0]
    dev_tmp = np.empty((0, n), int)
    for i in range(div_num):
        tmp = divide[i][:]
        for j in range(n):
            if (j == 0 or tmp[j] < tmp[j - 1]):
                tmp[j] = tmp[j] + 1
                dev_tmp = np.append(dev_tmp, tmp.reshape(1, n), axis=0)
                tmp[j] = tmp[j] - 1
    return (np.unique(dev_tmp, axis=0))

def divides(A, n):
    if (A == 0):
        return np.zeros((1, n))
    devs = np.zeros((1, n))
    devs[0][0] = 1
    for i in range(A - 1):
        devs_next = next_divide(devs)
        devs = devs_next
    return (devs)

def clash_matrix(s_A, s_B):
    assert s_A.shape == s_B.shape
    fields_number = s_A.shape[0]
    matrix = np.zeros((fields_number, fields_number))
    for i in range(fields_number):
        for j in range(fields_number):
            matrix[i,j] = s_A[j] - s_B[i]
    matrix = np.sign(matrix)
    return matrix

def k_W_and_k_L(s_A, s_B): #number of battlefields won and lost by A for those two pure strategies
    result = np.sign(s_A - s_B)
    k_W = (result==1).sum()
    k_L = (result==-1).sum()
    return k_W, k_L

def h(s_A, s_B): #Subsequent rows denote number of battlefields won by A, columns denobe the same for B
    fields_number = s_A.shape[0]
    values = np.zeros((fields_number+1, fields_number+1))
    s_B_permutations = np.array(list(set(permutations(s_B.tolist()))))
    for i in range(s_B_permutations.shape[0]):
        k_W, k_L = k_W_and_k_L(s_A, s_B_permutations[i])
        values[k_W, k_L] += 1
    values *= (math.factorial(fields_number) / values.sum())
    return values

def find_L_and_T(clash_matrix):
    fields_number = clash_matrix.shape[0]
    L = np.zeros((fields_number))
    T = np.zeros((fields_number))
    for i in range(fields_number):
        L[i] = (clash_matrix[:,i]==-1).sum()
        T[i] = (clash_matrix[:,i]==0).sum()
    return L, T

def find_knots(L, T):
    fields_number = L.shape[0]
    i = fields_number
    j = fields_number
    current_height = fields_number
    knots = np.array([[i,j]])
    while(L.shape[0] > 0):
        height_to_remove, width_to_remove = height_and_width_to_cutoff(L, T, current_height)
        i -= height_to_remove
        current_height -= height_to_remove
        j -= width_to_remove
        if(width_to_remove > 0):
            L = L[:-width_to_remove]
            T = T[:-width_to_remove]
        if(i > 0 and j > 0):
            knots = np.append(knots, np.array([[i,j]]).reshape(1,2), axis=0)
        else:
            break
    knots = np.flip(knots, axis=0)
    return knots

def height_and_width_to_cutoff(L, T, current_height):
    if(L[-1] + T[-1] < current_height): #Corner in W
        return int(current_height - L[-1] - T[-1]), int(0)
    if(L[-1] == current_height): #Corner in L
        width = (L == current_height).sum()
        return int(0), int(width)
    if(T[-1] > 0): #Corner in T
        if(np.all(T == T[-1]) and np.all(L == L[-1])): #Only two stripe (T and L) left
            return int(T[-1]), int(0)
        elif(np.all(L == 0) and T[0] < current_height): #Only two stripe (W and T) left
            width = (T == T[-1]).sum()
            return int(0), int(width)
        else:
            height = T[-1]
            width = min( (L == L[-1]).sum(), (L + T == L[-1] + T[-1]).sum())
            return int(height), int(width)
    print("height_and_width_to_cutoff FUNCTION FAILED TO RETURN VALUES")

def newton_symbol(n,k):
    return scipy.special.comb(n , k, exact=True)

def single_type_rectangle(cols_num, rows_num, rooks_num):
    if(cols_num < 0 or rows_num < 0 or rooks_num < 0):
        # print("INCORRECT CALL FOR single_type_rectangle FUNCTION, AT LEAST ONE ARGUMENT SMALLER THAN 0")
        return 0
    if(rooks_num > cols_num or rooks_num > rows_num):
        # print("INCORRECT CALL FOR single_type_rectangle FUNCTION, MORE ROOKS THAN POSSIBLE")
        return 0
    return newton_symbol(rows_num, rooks_num) * newton_symbol(cols_num, rooks_num) * math.factorial(rooks_num)

def recurrence_H(s_A, s_B):
    fields_number = s_A.shape[0]
    clash_mat = clash_matrix(s_A, s_B)
    L,T = find_L_and_T(clash_mat)
    knots = find_knots(L, T)
    values = np.zeros((knots.shape[0], fields_number + 1, fields_number + 1, fields_number + 1))  #knot index, number of rooks, k_W, k_L
    values[:,0,0,0] = 1
    # first knot, we know it exists
    knot = knots[0]
    i = knot[0]
    j = knot[1]
    if (L[0] > 0):  # first area in L
        for num_rooks in range(min(i, j) + 1):
            values[0, num_rooks, 0, num_rooks] = single_type_rectangle(i, j, num_rooks)
    elif (T[0] > 0):  # first area in T
        for num_rooks in range(min(i, j) + 1):
            values[0, num_rooks, 0, 0] = single_type_rectangle(i, j, num_rooks)
    else:  # first area in W
        for num_rooks in range(min(i, j) + 1):
            values[0, num_rooks, num_rooks, 0] = single_type_rectangle(i, j, num_rooks)
    for knot_index in range(1, knots.shape[0]-1):
        knot = knots[knot_index]
        i = knot[0]
        j = knot[1]
        previous_knot = knots[knot_index-1]
        old_i = previous_knot[0]
        old_j = previous_knot[1]
        delta_i = i - old_i
        delta_j = j - old_j
        if(knot_index == 1 and L[0] > 0 and T[0] > 0): # L and T stripes
            maximum_rooks_in_T = min(delta_i, j)
            for num_rooks in range(min(i, j) + 1):
                for r_T in range(min(maximum_rooks_in_T, num_rooks) + 1):
                    rooks_left = num_rooks - r_T
                    H_tmp = values[0, rooks_left, 0, rooks_left]
                    bottom = single_type_rectangle(delta_i, j - rooks_left, r_T)
                    values[knot_index, num_rooks, 0, rooks_left] = H_tmp * bottom
                    # print(num_rooks, r_T, rooks_left, H_tmp,  bottom)
        elif(knot_index == 1 and T[0] == 0 and T[j-1] > 0 and np.all(L[:j]==0)): # T and W stripes
            maximum_rooks_in_T = min(i, delta_j)
            for num_rooks in range(min(i, j) + 1):
                for r_T in range(min(maximum_rooks_in_T, num_rooks) + 1):
                    rooks_left = num_rooks - r_T
                    H_tmp = values[0, rooks_left, rooks_left, 0]
                    right = single_type_rectangle(i - rooks_left, delta_j, r_T)
                    values[knot_index, num_rooks, rooks_left, 0] = H_tmp * right
        else:
            maximum_rooks_in_L = min(old_i, delta_j)
            maximum_rooks_in_T = min(delta_i, delta_j)
            maximum_rooks_in_W = min(delta_i, old_j)
            for num_rooks in range(min(i, j) + 1):
                for k_W in range(num_rooks + 1):
                    for k_L in range(num_rooks + 1 - k_W):
                        sum = 0
                        for r_L in range(min(maximum_rooks_in_L, num_rooks) + 1):
                            for r_T in range(min(maximum_rooks_in_T, num_rooks) + 1):
                                for r_W in range(min(maximum_rooks_in_W, num_rooks) + 1):
                                    if(r_L + r_W + r_T <= num_rooks):
                                        rooks_left = num_rooks - r_W - r_T - r_L
                                        H_tmp = values[knot_index - 1, rooks_left, k_W - r_W, k_L-r_L]
                                        bottom = single_type_rectangle(delta_i, old_j - rooks_left, r_W)
                                        corner = single_type_rectangle(delta_i - r_W, delta_j, r_T)
                                        right = single_type_rectangle(old_i - rooks_left, delta_j - r_T, r_L)
                                        sum += H_tmp * bottom * corner * right
                        values[knot_index, num_rooks, k_W, k_L] = sum
    # last knot
    if(knots.shape[0] > 1):
        i = fields_number
        j = fields_number
        old_knot = knots[-2]
        old_i, old_j = old_knot
        delta_i = i - old_i
        delta_j = j - old_j
        maximum_rooks_in_L = min(old_i, delta_j)
        maximum_rooks_in_T = min(delta_i, delta_j)
        maximum_rooks_in_W = min(delta_i, old_j)
        num_rooks = fields_number
        for k_W in range(num_rooks + 1):
            for k_L in range(num_rooks + 1 - k_W):
                sum = 0
                for r_L in range(min(maximum_rooks_in_L, num_rooks) + 1):
                    for r_T in range(min(maximum_rooks_in_T, num_rooks) + 1):
                        for r_W in range(min(maximum_rooks_in_W, num_rooks) + 1):
                            if (r_L + r_W + r_T <= num_rooks):
                                rooks_left = num_rooks - r_W - r_T - r_L
                                H_tmp = values[-2, rooks_left, k_W - r_W, k_L - r_L]
                                bottom = single_type_rectangle(delta_i, old_j - rooks_left, r_W)
                                corner = single_type_rectangle(delta_i - r_W, delta_j, r_T)
                                right = single_type_rectangle(old_i - rooks_left, delta_j - r_T, r_L)
                                sum += H_tmp * bottom * corner * right
                values[-1, num_rooks, k_W, k_L] = sum
    return values

def symmetrized_payoff(s_A, s_B, aggregation_function):
    fields_number = s_A.shape[0]
    h = recurrence_H(s_A, s_B)[-1,-1,:,:]
    result = 0
    for k_W in range(h.shape[0]):
        for k_L in range(h.shape[1]):
            result += h[k_W, k_L] * aggregation_function(k_W, k_L, fields_number)
    result /= math.factorial(fields_number)
    return result

def symmetrized_payoff_parralel(pack):
    i, j, s_A, s_B, aggregation_function = pack
    fields_number = s_A.shape[0]
    h = recurrence_H(s_A, s_B)[-1,-1,:,:]
    result = 0
    for k_W in range(h.shape[0]):
        for k_L in range(h.shape[1]):
            result += h[k_W, k_L] * aggregation_function(k_W, k_L, fields_number)
    result /= math.factorial(fields_number)
    return i, j, result

def payoff_matrix_pandas(A, B, fields_number, aggregation_function):
    A_strategies = divides(A, fields_number)
    B_strategies = divides(B, fields_number)
    matrix = np.zeros((A_strategies.shape[0], B_strategies.shape[0]))
    for A_index in range(A_strategies.shape[0]):
        for B_index in range(B_strategies.shape[0]):
            matrix[A_index, B_index] = symmetrized_payoff(A_strategies[A_index], B_strategies[B_index], aggregation_function)
    columns_names, rows_names = get_columns_and_rows_names(A_strategies, B_strategies)
    matrix = pd.DataFrame(matrix, columns=columns_names, index=rows_names)
    return matrix

def payoff_matrix_pandas_multithreading(A, B, fields_number, aggregation_function, threads_number=None):
    A_strategies = divides(A, fields_number)
    B_strategies = divides(B, fields_number)
    matrix = np.zeros((A_strategies.shape[0], B_strategies.shape[0]))
    args = ((i, j, A_strategies[i], B_strategies[j], aggregation_function) for i in range(A_strategies.shape[0]) for j in range(B_strategies.shape[0]))
    with concurrent.futures.ProcessPoolExecutor(max_workers=threads_number) as executor:
        for A_index, B_index, val in executor.map(symmetrized_payoff_parralel, args):
            matrix[A_index, B_index] = val
    columns_names, rows_names = get_columns_and_rows_names(A_strategies, B_strategies)
    matrix = pd.DataFrame(matrix, columns=columns_names, index=rows_names)
    return matrix

def get_columns_and_rows_names(A_strategies, B_strategies):
    columns_names = []
    rows_names = []
    for i in range(A_strategies.shape[0]):
        rows_names.append(str(A_strategies[i]))
    for i in range(B_strategies.shape[0]):
        columns_names.append(str(B_strategies[i]))
    return columns_names, rows_names

def blotto(k_W, k_L, n):
    return k_W - k_L

def attack(k_W, k_L, n):
    if(k_W > 0):
        return 1
    return -1

def chopstick(k_W, k_L, n):
    return np.sign(k_W - k_L)

def majoritarian(k_W, k_L, n):
    if(k_W > n/2):
        return 1
    if(k_L > n/2):
        return -1
    return 0