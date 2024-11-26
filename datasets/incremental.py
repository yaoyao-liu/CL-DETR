import numpy as np

def generate_cls_order(num_of_cls=91, seed=123):
    """The function to create the class order for incremental learning.
    Arg:
        num_of_cls: the number of classes.
        seed: the random seed.
    """
    # Create class order
    np.random.seed(seed)
    cls_order = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
    np.random.shuffle(cls_order)
    print('creating the training class order...')
    print('current random seed: ' + str(seed))
    print('current class order: ' + str(cls_order))

    return cls_order