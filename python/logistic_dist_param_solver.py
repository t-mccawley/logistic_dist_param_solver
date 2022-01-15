import math
import numpy as np

# constants
CHALLENGER_CDF = 0.65
RIVAL_CDF = 0.9
DUELIST_CDF = 0.97
GLADIATOR_CDF = 0.995
RANK_1_CDF = 0.999
MU_STEP = 1
S_MIN = 1
S_MAX = 1000
S_STEP = 1


# input title ratings
challenger_rating = 1400
rival_rating = 1700
duelist_rating = 2000
gladiator_rating = 2600
rank_1_rating = 2800
ratings_cdf_dict = {
    challenger_rating:CHALLENGER_CDF,
    rival_rating:RIVAL_CDF,
    duelist_rating:DUELIST_CDF,
    gladiator_rating:GLADIATOR_CDF,
    rank_1_rating:RANK_1_CDF,
    }

def logistic_distribution_cdf(x,mu,s):
    """CDF of logistic distrbution"""
    return( 1 / (1 + math.exp( (mu - x) / s ) ) )

def logistic_distribution_cdf_abs_error(x,mu,s,p):
    """Error of CDF given p"""
    return(abs(p - logistic_distribution_cdf(x,mu,s)))

def arena_rating_cdf_error(mu,s,ratings_cdf_dict):
    """Given CDF parameters and arena title ratings, compute the absolute error"""
    error = 0
    for rating in ratings_cdf_dict:
        error += logistic_distribution_cdf_abs_error(rating,mu,s,ratings_cdf_dict[rating])
    return(error)

def logistic_dist_param_solve(ratings_cdf_dict,challenger_rating):
    """Solve for the parameters which minimize the error of the arena rating logistic distribution"""
    error_sol = np.inf
    mu_sol = None
    s_sol = None
    for mu in np.arange(0,challenger_rating,MU_STEP):
        for s in np.arange(S_MIN,S_MAX,S_STEP):
            error = arena_rating_cdf_error(mu,s,ratings_cdf_dict)
            if error < error_sol:
                error_sol = error
                mu_sol = mu
                s_sol = s
    return(mu_sol,s_sol,error_sol)

# solve
mu,s,error = logistic_dist_param_solve(ratings_cdf_dict,challenger_rating)

print("mu: {:0.0f}, s: {:0.0f}, error: {:0.5f}".format(mu,s,error))