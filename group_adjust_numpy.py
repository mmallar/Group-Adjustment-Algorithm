'''
    group_adjust_numpy.py
    
	Author: Michael Mallar
    Date created: 11/06/2018
    Date last modified: 11/07/2018
    Python Version: 3.6.5
	
	To Run:
		1. Install pytest
		2. Run 'pytest group_adjust_numpy'
'''

import pytest
from datetime import datetime
import numpy as np

# Code Challenge:
# Write the group adjustment method below. 
# Solution can be pure python, pure NumPy, pure Pandas, or Combination  

# Group Adjust Method
# The algorithm needs to do the following:
# 1.) For each group-list provided, calculate the means of the values for each
# 	  unique group.
#
#   For example:
#   	vals       = [  1  ,   2  ,   3  ]
#   	ctry_grp   = ['USA', 'USA', 'USA']
#   	state_grp  = ['MA' , 'MA' ,  'CT' ]
#
#   There is only 1 country in the ctry_grp list.  So to get the means:
#     	USA_mean == mean(vals) == 2
#     	ctry_means = [2, 2, 2]
#   There are 2 states, so to get the means for each state:
#     	MA_mean == mean(vals[0], vals[1]) == 1.5
#     	CT_mean == mean(vals[2]) == 3
#     	state_means = [1.5, 1.5, 3]
#
# 2.) Using the weights, calculate a weighted average of those group means
#	
#	Continuing from our example:
#  		weights = [.35, .65]
#   	35% weighted on country, 65% weighted on state
#   	ctry_means  = [2  , 2  , 2]
#   	state_means = [1.5, 1.5, 3]
#   	weighted_means = [2*.35 + .65*1.5, 2*.35 + .65*1.5, 2*.35 + .65*3]
#
# 3.) Subtract the weighted average group means from each original value
#   
#	Continuing from our example:
#   	val[0] = 1
#   	ctry[0] = 'USA' --> 'USA' mean == 2, ctry weight = .35
#   	state[0] = 'MA' --> 'MA'  mean == 1.5, state weight = .65
#   	weighted_mean = 2*.35 + .65*1.5 = 1.675
#   	demeaned = 1 - 1.675 = -0.675
#   
#	Do this for all values in the original list.
#
# 4.) Return the demeaned values


# Creating a hash table for the input groups
def generate_data_dict(init_data,unique_data):
    elem_dict = {}
    for elem in unique_data:
        col_idxs = np.unique(np.where(init_data == elem)[1])
        row_idx = np.unique(np.where(init_data == elem)[0])
        elem_count = len(col_idxs)
        elem_dict.update({
            elem:{
                'col-idxs':col_idxs,
                'row-idx':row_idx,
                'count':elem_count
            }
        })
    return elem_dict


# Calculating weighted mean values and generating a hash table 
def generate_weight_mean_dict(elem_values,data_dict,group_weights):
    weighted_dict = {}
    for elem in data_dict.keys():
        val_idxs = data_dict[elem]['col-idxs']
        weight_idx = data_dict[elem]['row-idx']
        mean = np.nanmean(elem_values[val_idxs])
        weighted_mean = mean*group_weights[weight_idx]
        weighted_dict.update({
            elem:weighted_mean.item(0)
        })
    return weighted_dict


# Replacing keys with respective values and stacking the groups 
def generate_weighted_stack(stack,weighted_dict):
    for key,value in weighted_dict.items():
        # where condition is true replace index with value else leave it alone
        stack = np.where(stack == key,value,stack)
    return stack


def group_adjust(vals, groups, weights):
    # parameter error checking 
    if not vals or not groups or not weights:
        raise ValueError("Parameter is missing.")
    
    if len(groups) != len(weights):
        raise ValueError("Dimenstion Error: groups: " + str(len(groups)) + "weights: " + str(len(weights)))
    
    if len(groups[0]) != len(vals):
        raise ValueError("Dimension Error: group: " + str(len(groups[0])) + "vals: " + str(len(vals)))
    
    init_data = np.asarray(groups)
    
    # get all unique elems of group
    unique_data = np.unique(init_data)
   
    # create a dictionary from groups data
    data_dict = generate_data_dict(init_data,unique_data)
    
    elem_values = np.asarray(vals)
    group_weights = np.asarray(weights)

    # create a dictionary of weights per unique element from groups
    weighted_dict = generate_weight_mean_dict(elem_values,data_dict,group_weights)
    
    # stack data from original input
    stack = np.stack(groups)
    
    # replace all the keys with their respective weight values
    stack = generate_weighted_stack(stack,weighted_dict)
    
    # transpose the stack
    stack = stack.T
    
    # set the data type 
    stack = stack.astype(np.float)
    
    #sum each row in the stack
    weighted_means = stack.sum(axis=1)
    
    # generate the demeaned list
    demeaned = elem_values - weighted_means

    return demeaned

# Pytests 
def test_three_groups():
    vals = [1, 2, 3, 8, 5]
    grps_1 = ['USA', 'USA', 'USA', 'USA', 'USA']
    grps_2 = ['MA', 'MA', 'MA', 'RI', 'RI']
    grps_3 = ['WEYMOUTH', 'BOSTON', 'BOSTON', 'PROVIDENCE', 'PROVIDENCE']
    weights = [.15, .35, .5]

    adj_vals = group_adjust(vals, [grps_1, grps_2, grps_3], weights)
    # 1 - (USA_mean*.15 + MA_mean * .35 + WEYMOUTH_mean * .5)
    # 2 - (USA_mean*.15 + MA_mean * .35 + BOSTON_mean * .5)
    # 3 - (USA_mean*.15 + MA_mean * .35 + BOSTON_mean * .5)
    # etc ...
    # Plug in the numbers ...
    # 1 - (.15*2 + .35*2 + .5*1)   # -0.5
    # 2 - (.15*2 + .35*2 + .5*2.5) # -.25
    # 3 - (.15*2 + .35*2 + .5*2.5) # 0.75
    # etc...

    answer = [-0.770, -0.520, 0.480, 1.905, -1.095]
    for ans, res in zip(answer, adj_vals):
        assert abs(ans - res) < 1e-5


def test_two_groups():
    vals = [1, 2, 3, 8, 5]
    grps_1 = ['USA', 'USA', 'USA', 'USA', 'USA']
    grps_2 = ['MA', 'RI', 'CT', 'CT', 'CT']
    weights = [.65, .35]

    adj_vals = group_adjust(vals, [grps_1, grps_2], weights)
    # 1 - (.65*2 + .35*1)   # -0.65
    # 2 - (.65*2 + .35*2.5) # -.175
    # 3 - (.65*2 + .35*2.5) # -.825
    answer = [-1.81999, -1.16999, -1.33666, 3.66333, 0.66333]
    for ans, res in zip(answer, adj_vals):
        assert abs(ans - res) < 1e-5


def test_missing_vals():
    # If you're using NumPy or Pandas, use np.NaN
    # If you're writing pyton, use None
    vals = [1, np.NaN, 3, 5, 8, 7]
    # vals = [1, None, 3, 5, 8, 7]
    grps_1 = ['USA', 'USA', 'USA', 'USA', 'USA', 'USA']
    grps_2 = ['MA', 'RI', 'RI', 'CT', 'CT', 'CT']
    weights = [.65, .35]

    adj_vals = group_adjust(vals, [grps_1, grps_2], weights)

    # This should be None or np.NaN depending on your implementation
    # please feel free to change this line to match yours
    answer = [-2.47, np.NaN, -1.170, -0.4533333, 2.54666666, 1.54666666]
    # answer = [-2.47, None, -1.170, -0.4533333, 2.54666666, 1.54666666]

    for ans, res in zip(answer, adj_vals):
        if ans is None:
            assert res is None
        elif np.isnan(ans):
            assert np.isnan(res)
        else:
            assert abs(ans - res) < 1e-5


def test_weights_len_equals_group_len():
    # Need to have 1 weight for each group

    vals = [1, np.NaN, 3, 5, 8, 7]
    #vals = [1, None, 3, 5, 8, 7]
    grps_1 = ['USA', 'USA', 'USA', 'USA', 'USA', 'USA']
    grps_2 = ['MA', 'RI', 'RI', 'CT', 'CT', 'CT']
    weights = [.65]

    with pytest.raises(ValueError):
        group_adjust(vals, [grps_1, grps_2], weights)


def test_group_len_equals_vals_len():
    # The groups need to be same shape as vals
    vals = [1, None, 3, 5, 8, 7]
    grps_1 = ['USA']
    grps_2 = ['MA', 'RI', 'RI', 'CT', 'CT', 'CT']
    weights = [.65]

    with pytest.raises(ValueError):
        group_adjust(vals, [grps_1, grps_2], weights)


def test_performance():
    # vals = 1000000*[1, None, 3, 5, 8, 7]
    # If you're doing numpy, use the np.NaN instead
    vals = 1000000 * [1, np.NaN, 3, 5, 8, 7]
    grps_1 = 1000000 * [1, 1, 1, 1, 1, 1]
    grps_2 = 1000000 * [1, 1, 1, 1, 2, 2]
    grps_3 = 1000000 * [1, 2, 2, 3, 4, 5]
    weights = [.20, .30, .50]

    start = datetime.now()
    group_adjust(vals, [grps_1, grps_2, grps_3], weights)
    end = datetime.now()
    diff = end - start
    print('Total performance test time: {}'.format(diff.total_seconds()))
