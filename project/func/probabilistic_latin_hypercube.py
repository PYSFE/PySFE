import numpy as np
import copy


def latin_hypercube_sampling(num_samples, num_arguments=1, sample_min=0, sample_max=1):

    if sample_min > sample_max:
        sample_max += sample_min
        sample_min = sample_max - sample_min
        sample_max = sample_max - sample_min

    # Generate sorted integers with correct shape
    list_random_num = []
    for i in range(num_arguments):
        mat_single_sorted = np.linspace(sample_min, sample_max, num_samples+1, dtype=float)
        band_width = mat_single_sorted[1] - mat_single_sorted[0]
        mat_single_sorted = mat_single_sorted[0:-1] + band_width * 0.5
        np.random.shuffle(mat_single_sorted)
        list_random_num.append(copy.copy(mat_single_sorted))

    mat_random_nums = np.concatenate(list_random_num).reshape((num_samples, num_arguments), order="F")

    if num_arguments == 1:
        result = mat_random_nums.flatten()
    else:
        result = mat_random_nums

    return result


if __name__ == "__main__":
    samples = 100
    arguments = 3

    res = latin_hypercube_sampling(samples, arguments, 0, 0.001)
    print(res)