#!/usr/bin/python
import math

def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    ### your code goes here

    # zip packes up all objects into a tuple - list
    cleaned_data = zip(ages, net_worths, abs(net_worths-predictions))
    # sort based on error
    cleaned_data = sorted(cleaned_data, key=lambda x: (x[2]))
    # calculate the number of elements that need to drop
    cleaned_data_count = int(math.ceil(len(cleaned_data)*0.9))
    # slice
    cleaned_data = cleaned_data[:cleaned_data_count]

    return cleaned_data

