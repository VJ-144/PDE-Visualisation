import numpy as np
import random


def main():

    mat = np.array([[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15]])
    print('Mat')
    print(mat)


    a = np.roll(mat, shift=1, axis=1)
    print('\nShifted')
    print(a)


    return 0


main()