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
your_team_rating = 2004
challenger_rating = 1422
rival_rating = 1857
duelist_rating = 2205
gladiator_rating = 2804
rank_1_rating = 3024
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
print("Solving for parameters of logistic distribution of arena ratings")
print()
print("Input Ratings:")
print("\tYour Team Rating: {}".format(your_team_rating))
print("\tChallenger: {}".format(challenger_rating))
print("\tRival: {}".format(rival_rating))
print("\tDuelist: {}".format(duelist_rating))
print("\tGladiator: {}".format(gladiator_rating))
print("\tRank 1: {}".format(rank_1_rating))
print()
mu,s,error = logistic_dist_param_solve(ratings_cdf_dict,challenger_rating)
print("Outputs:")
print("\tmu: {:0.0f}".format(mu))
print("\ts: {:0.0f}".format(s))
print("\terror: {:0.5f}".format(error))
print("\tYour Team Percentile: {:0.1f}th".format(100*logistic_distribution_cdf(your_team_rating,mu,s)))