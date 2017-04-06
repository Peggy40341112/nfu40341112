from probability_0406 import normal_cdf, inverse_normal_cdf
import math, random

def normal_approximation_to_binomial(n, p):
    """finds mu and sigma corresponding to a Binomial(n, p)"""
    mu = p * n
    sigma = math.sqrt(p * (1 - p) * n)
    return mu, sigma

def normal_upper_bound(probability, mu=0, sigma=1):
    """returns the z for which P(Z <= z) = probability"""
    return inverse_normal_cdf(probability, mu, sigma)

def normal_lower_bound(probability, mu=0, sigma=1):
    """returns the z for which P(Z >= z) = probability"""
    return inverse_normal_cdf(1 - probability, mu, sigma)

def normal_two_sided_bounds(probability, mu=0, sigma=1):
    """returns the symmetric (about the mean) bounds
    that contain the specified probability"""
    tail_probability = (1 - probability) / 2

    # upper bound should have tail_probability above it
    upper_bound = normal_lower_bound(tail_probability, mu, sigma)

    # lower bound should have tail_probability below it
    lower_bound = normal_upper_bound(tail_probability, mu, sigma)



if __name__ == "__main__":

    p=0.99
    a=0.46
    mu_0, sigma_0 = normal_approximation_to_binomial(1000, a)
    print("mu_0", mu_0)
    print("sigma_0", sigma_0)
    print("normal_two_sided_bounds("+str(p)+", mu_0, sigma_0)", normal_two_sided_bounds(p, mu_0, sigma_0))
    print

    p=0.95
    a=0.46
    mu_0, sigma_0 = normal_approximation_to_binomial(1000, a)
    print("mu_0", mu_0)
    print("sigma_0", sigma_0)
    print("normal_two_sided_bounds("+str(p)+", mu_0, sigma_0)", normal_two_sided_bounds(p, mu_0, sigma_0))
    print

    p=0.9
    a=0.46
    mu_0, sigma_0 = normal_approximation_to_binomial(1000, a)
    print("mu_0", mu_0)
    print("sigma_0", sigma_0)
    print("normal_two_sided_bounds("+str(p)+", mu_0, sigma_0)", normal_two_sided_bounds(p, mu_0, sigma_0))
    print

