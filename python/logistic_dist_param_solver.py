import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict

# constants
CHALLENGER_CDF = 0.65
RIVAL_CDF = 0.9
DUELIST_CDF = 0.97
GLADIATOR_CDF = 0.995
RANK_ONE_CDF = 0.999
MU_STEP = 1
S_MIN = 1
S_MAX = 1000
S_STEP = 1
BRACKETS = ['2v2','3v3','5v5']
RATING_PLOT_MIN = 0
RATING_PLOT_MAX = 3000
BRACKET_COLOR_MAP = {
    '2v2':'b',
    '3v3':'y',
    '5v5':'r',
}

# parse inputs.csv
inputs = pd.read_csv('../inputs/inputs.csv')
inputs = inputs.set_index('bracket')

# construct dicts
ratings_cdf_dict = {}
for bracket in BRACKETS:
    ratings_cdf_dict[bracket] = {
    inputs.loc[bracket,'challenger']:CHALLENGER_CDF,
    inputs.loc[bracket,'rival']:RIVAL_CDF,
    inputs.loc[bracket,'duelist']:DUELIST_CDF,
    inputs.loc[bracket,'gladiator']:GLADIATOR_CDF,
    inputs.loc[bracket,'rank_one']:RANK_ONE_CDF,
    }
ratings_title_dict = {}
for bracket in BRACKETS:
    ratings_title_dict[bracket] = OrderedDict()
    ratings_title_dict[bracket][inputs.loc[bracket,'rank_one']] = 'Rank One'
    ratings_title_dict[bracket][inputs.loc[bracket,'gladiator']] = 'Gladiator'
    ratings_title_dict[bracket][inputs.loc[bracket,'duelist']] = 'Duelist'
    ratings_title_dict[bracket][inputs.loc[bracket,'rival']] = 'Rival'
    ratings_title_dict[bracket][inputs.loc[bracket,'challenger']] = 'Challenger'

def logistic_distribution_cdf_fast(x,mu,s):
    """CDF of logistic distrbution"""
    return( 1 / (1 + math.exp( (mu - x) / s ) ) )

def logistic_distribution_cdf(x,mu,s):
    """CDF of logistic distrbution"""
    return( 1 / (1 + np.exp( (mu - x) / s ) ) )

def logistic_distribution_cdf_abs_error(x,mu,s,p):
    """Error of CDF given p"""
    return(abs(p - logistic_distribution_cdf_fast(x,mu,s)))

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

def determine_title(player_rating,ratings_title_dict,bracket):
    for rating in ratings_title_dict[bracket]:
        if player_rating >= rating:
            return(ratings_title_dict[bracket][rating])
    return('None')

# solve
outputs = pd.DataFrame(index=BRACKETS,columns=['mu','s','error','player_percentile'])
print("Solving for parameters of logistic distribution of arena ratings")
print()
for bracket in BRACKETS:
    print('Bracket: {}'.format(bracket))
    print("Inputs:")
    print(inputs.loc[bracket,:])
    print()
    mu,s,error = logistic_dist_param_solve(ratings_cdf_dict[bracket],inputs.loc[bracket,'challenger'])
    player_percentile = 100*logistic_distribution_cdf_fast(inputs.loc[bracket,'player_rating'],mu,s)
    outputs.loc[bracket,'mu'] = mu
    outputs.loc[bracket,'s'] = s
    outputs.loc[bracket,'error'] = error
    outputs.loc[bracket,'player_percentile'] = player_percentile
    print("Outputs:")
    print(outputs.loc[bracket,:])
    print()


# visualize
rating_plot = np.arange(RATING_PLOT_MIN,RATING_PLOT_MAX,1)

plt.figure(figsize=(14,7))

for bracket in BRACKETS:
    cdf_plot = 100*logistic_distribution_cdf(rating_plot,outputs.loc[bracket,'mu'],outputs.loc[bracket,'s'])
    bracket_color = BRACKET_COLOR_MAP[bracket]
    plt.plot(rating_plot,cdf_plot,label=bracket,color=bracket_color)
    player_rating = inputs.loc[bracket,'player_rating']
    player_percentile = outputs.loc[bracket,'player_percentile']
    player_title = determine_title(player_rating,ratings_title_dict,bracket)
    player_str = 'Rating = {}, Percentile = {:0.2f}, Title = {}'.format(player_rating,player_percentile,player_title)
    plt.plot(player_rating,player_percentile,label=player_str,marker='o',color=bracket_color)
    # plot title cutoffs 
    # TODO

plt.legend()
plt.title('WoW Arena Rating CDFs')
plt.xlabel('Rating')
plt.ylabel('Percentile')
plt.xticks(np.arange(RATING_PLOT_MIN, RATING_PLOT_MAX+1, step=100))
plt.xticks(rotation = 45)
plt.yticks(np.arange(0, 101, step=5))
plt.grid()
plt.show()