import math

def weighted_mean(means_and_stds_and_freqs):
    total_weight = 0
    means_and_pfds_and_freqs = []
    for mean_and_std_and_freq in means_and_stds_and_freqs:
        mean = mean_and_std_and_freq[0]
        std = mean_and_std_and_freq[1]
        freq = mean_and_std_and_freq[2]
        total_weight += freq
        means_and_pfds_and_freqs.append([mean, probabalistic_density_function_of_mean(std), freq])

    # Weighted mean = Sum(pfd * mean) / Sum(pfd)
    numerator = 0
    denominator = 0
    running_total = 0
    for mean_and_pfd_and_freq in means_and_pfds_and_freqs:
        mean = mean_and_pfd_and_freq[0]
        pfd = mean_and_pfd_and_freq[1]
        freq = mean_and_pfd_and_freq[2]
        running_total += (freq / total_weight) * mean
        numerator += mean * pfd
        denominator += pfd

    return 0 if numerator == 0 else (numerator / denominator)


def probabalistic_density_function_of_mean(std):
    return 1 / math.sqrt(2 * math.pi * std * std)

# Valence is between 1 and 9. To reverse, we flip over 5:
# e.g. 1 becomes 9, and 7.5 becomes 3.5
def reverse_valence(val):
    return 5 - (val - 5)