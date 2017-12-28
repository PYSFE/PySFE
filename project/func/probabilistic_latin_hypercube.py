import numpy as np
import copy


def latin_hypercube_sampling(num_samples, num_arguments=1, sample_min=0, sample_max=1):

    if sample_min > sample_max:
        sample_max += sample_min
        sample_min = sample_max - sample_min
        sample_max = sample_max - sample_min

    # Generate sorted integers with correct shape
    mat_random_num = np.linspace(sample_min, sample_max, num_samples+1, dtype=float)
    mat_random_num += (mat_random_num[1] - mat_random_num[0]) * 0.5
    mat_random_num = mat_random_num[0:-1]
    mat_random_num = np.reshape(mat_random_num, (len(mat_random_num), 1))
    mat_random_nums = mat_random_num * np.ones((1, num_arguments))

    # np.random.shuffle(mat_random_nums)

    for i in range(np.shape(mat_random_nums)[1]):
        np.random.shuffle(mat_random_nums[:,i])

    if num_arguments == 1:
        mat_random_nums = mat_random_nums.flatten()

    return mat_random_nums


if __name__ == "__main__":
    samples = 100
    arguments = 3

    res = latin_hypercube_sampling(samples, arguments, 0, 0.001)
    print(res)