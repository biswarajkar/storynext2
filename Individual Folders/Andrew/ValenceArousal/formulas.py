import math

def weighted_mean(means_and_stds):
    means_and_pfds = []
    for mean_and_std in means_and_stds:
        mean = mean_and_std[0]
        std = mean_and_std[1]
        means_and_pfds.append([mean, probabalistic_density_function_of_mean(std)])

    # Weighted mean = Sum(pfd * mean) / Sum(pfd)
    numerator = 0
    denominator = 0
    for mean_and_pfd in means_and_pfds:
        mean = mean_and_pfd[0]
        pfd = mean_and_pfd[1]
        numerator += mean * pfd
        denominator += pfd

    return 0 if numerator == 0 else (numerator / denominator)


def probabalistic_density_function_of_mean(std):
    return 1 / math.sqrt(2 * math.pi * std)