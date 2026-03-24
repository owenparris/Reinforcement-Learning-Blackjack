import numpy as np

def create_strategy_arrays():

    split_array = np.zeros((12,12))
    double_array_soft = np.zeros((22,12))
    double_array_hard = np.zeros((22,12))
    stand_array_soft = np.zeros((22,12))
    stand_array_hard = np.zeros((22,12))
    
    for i in (1, 8):
        for j in range(1,11):
            split_array[i][j] = 1
    for i in (2,3,7):
        for j in range(2,8):
            split_array[i][j] = 1
    
    for j in range(2,7):
        split_array[6][j] = 1
    for j in (5,6):
        split_array[4][j] = 1
    for j in (2,3,4,5,6,8,9):
        split_array[9][j] = 1

    for i in (17,18):
        for j in range(3,7):
            double_array_soft[i][j]= 1
    for i in (15,16):
        for j in range(4,7):
            double_array_soft[i][j]= 1
    for i in (13,14):
        for j in (5,6):
            double_array_soft[i][j]= 1
    

    for j in range(2,11):
        double_array_hard[11][j] = 1
    for j in range(2,10):
        double_array_hard[10][j] = 1
    for j in range(3,7):
        double_array_hard[9][j] = 1
    
    for i in (19,20,21):
        for j in range(1,11):
            stand_array_soft[i][j] = 1
    for j in range(2,9):
        stand_array_soft[18][j] = 1
    
    for i in range(17,22):
        for j in range(1,11):
            stand_array_hard[i][j] = 1
    for i in range(13,17):
        for j in range(2,7):
            stand_array_hard[i][j] = 1
    for j in range(4,7):
        stand_array_hard[12][j] = 1

    np.save('strategy_arrays/split_array.npy', split_array)
    np.save('strategy_arrays/double_array_soft.npy', double_array_soft)
    np.save('strategy_arrays/double_array_hard', double_array_hard)
    np.save('strategy_arrays/stand_array_soft.npy', stand_array_soft)
    np.save('strategy_arrays/stand_array_hard.npy', stand_array_hard)
    
create_strategy_arrays()