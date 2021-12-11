import numpy as np
from itertools import permutations
import math
import scipy.special

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
test_H(10, 3)
#%%

def test_knots(A, n):
    strats = divides(A, n)
    for i in range(strats.shape[0]):
        for j in range(strats.shape[0]):
            mock_A = strats[i]
            mock_B = strats[j]
            clash_mat = clash_matrix(mock_A, mock_B)
            L, T = find_L_and_T(clash_mat)
