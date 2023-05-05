import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import TrAdaBoost
import kmm
from sklearn.metrics import pairwise_distances
from scipy.stats import wasserstein_distance
import time
from multiprocessing import Process, Queue

def execute_sds_and_ntm(training_set, source_domain_list, ntm_method):

    '''
    chooses the best fitting sourcedomains according to the wasserstein metrik and weights the sampels with the choosen ntm methode

    :param training_set: dataframe including all domains/ column 'source_domain' contains the informations which source domain the sample comes from

    :param source_domain_list: list of all 'source_domain' names 
    
    :param ntm_method: method to calculate sample weights must be "kmm" or "tradaboost"

    :return: training set and sample weights for model application 
    '''

    # set target domain

    df_tar = training_set[training_set['source_domain'] == "target"]

    # get best fitting domains and calculate sample weights

    if len(df_tar) != 0:
                
        ### get best fitting source domains ###
        
        was_list__source_domain = []

        # calculate wasserstein distance for each market combination

        for i in source_domain_list:
            df_src = training_set[training_set['source_domain'] == i]
            df_src.fillna(0, inplace=True)

            if len(df_src) == 0:
                was = np.nan
            else: 
                was = wasserstein_distance(df_src["rv"],df_tar["rv"])

            was_list__source_domain.append(was)
    
        was_list_len = []

        # get lenth of each market

        for i in source_domain_list:

            df_src = training_set[training_set['source_domain'] == i]
            df_src.fillna(0, inplace=True)

            was_list_len.append(len(df_src))

        # put results into a df and calculate weight

        d = {'was': was_list__source_domain, 'len': was_list_len, 'source_domain': source_domain_list, 'weight': was_list__source_domain * (was_list_len-np.sum(was_list_len))*-1}
        df_was_markets = pd.DataFrame(data=d)
        df_was_markets = df_was_markets.sort_values(by=['weight'])
        df_was_markets.fillna(0, inplace=True)
        print(df_was_markets)

        # get source domains with min weights distance to target market

        df_src = pd.DataFrame()

        another_market = 1
        
        while another_market <= 4 and df_was_markets.iloc[another_market]['weight'] < df_was_markets.iloc[1]['weight']*1.2:
            df_src_2nd_market = training_set[training_set['source_domain'] == df_was_markets.iloc[another_market]['source_domain']]
            df_src = pd.concat([df_src, df_src_2nd_market])
            print("Another source domain was added: "+ df_was_markets.iloc[another_market]['source_domain'])
            another_market += 1
        
        # calculate weights for each sample and set market as source dataset
        
        training_set = pd.concat([df_src, df_tar]) # overwrite with additional source domains that fit target market
        training_set.fillna(0, inplace=True)

        if ntm_method == "kmm":

            kmm_weighting(training_set)

        elif ntm_method == "tradaboost":

            beta = tradaboost_weighting(df_src, df_tar)

        else:
            print("Please choose a sample weighting algorithm")

    else:

        # if target is 0 weights choosen standard weights

        beta = np.array ( [1]*len(training_set) ) # all weights are the same

    return training_set, beta



def tradaboost_weighting(df_src, df_tar): 

    '''
    Calculate sample weights with TrAdaBoost

    :param df_src: dataframe including all source domain sampels

    :param df_tar: dataframe including all target domain sampels

    :return beta: sampel weight the length of src+tar
    '''

    try:

        # prepare Dataset

        n_source = len(df_src)
        x_source = df_src.iloc[:,:-1]._get_numeric_data()
        y_source = df_src.iloc[:,-1]._get_numeric_data()

        n_target = len(df_tar)
        x_target = df_tar.iloc[:,:-1]._get_numeric_data()
        y_target = df_tar.iloc[:,-1]._get_numeric_data()

        X = np.concatenate((x_source, x_target))
        y = np.concatenate((y_source, y_target))
        sample_size = [n_source, n_target]

        # define Hyperparameters

        n_estimators = 80
        steps = 10
        fold = 5
        random_state = np.random.RandomState(1)

        regr = TrAdaBoost.TwoStageTrAdaBoostR2( DecisionTreeRegressor(max_depth=4),
                            n_estimators = n_estimators, sample_size = sample_size, 
                            steps = steps, fold = fold, 
                            random_state = random_state)

        regr_test = TrAdaBoost.TwoStageTrAdaBoostR2( DecisionTreeRegressor(max_depth=4),
                            n_estimators = n_estimators, sample_size = sample_size, 
                            steps = 1, fold = fold, 
                            random_state = random_state)

        q = Queue()

        # Start process from multiprocesses

        p = Process(target=regr_test.fit, args=(X, y, None, q))
        p.start()

        # Wait 10 seconds for the process

        time.sleep(10)

        # Check if process is still running

        if p.is_alive():
            print ("process is still running, it will be terminated now")

            # Terminate process

            p.terminate()
            p.join()

            # set standard weights by using the cosine distance as backup

            matrix = pairwise_distances(training_set._get_numeric_data(),df_tar._get_numeric_data(), metric='cosine')
            df_matrix = pd.DataFrame(matrix)
            df_matrix = df_matrix.mean(axis=1)
            beta = df_matrix

        else:

            # get return value from process with 10 steps

            p = Process(target=regr.fit, args=(X, y, None, q))
            p.start()
            beta = q.get()
            beta = beta[1]

    except MemoryError as err: 
        print("Error: ", err)
        beta = np.array ( [1]*len(training_set) )
        print("Beta is assigned", beta)
        pass
    
    return beta


def kmm_weighting(training_set):

    '''
    Calculate sample weights with KMM

    :param training_set: dataframe including all target and source domain sampels

    :return beta: sampel weight the length of src+tar
    '''


    try: 
        kmm_1 = kmm.KMM()

        q = Queue()
        
        # Start process from multiprocesses

        p = Process(target=kmm_1.fit, args=(training_set._get_numeric_data(), df_tar._get_numeric_data(), 1, q))
        p.start()

        # Wait 2 seconds for the process to test wrather 2 iterations need less than 2 seconds

        time.sleep(2)

        # Check if process is still running

        if p.is_alive():

            print ("process is still running, it will be terminated now")

            # Terminate process

            p.terminate()
            p.join()

            # set standard weights by using the cosine distance as backup

            matrix = pairwise_distances(training_set._get_numeric_data(),df_tar._get_numeric_data(), metric='cosine')
            df_matrix = pd.DataFrame(matrix)
            df_matrix = df_matrix.mean(axis=1)
            beta = df_matrix

        else:

            # get return value from whole process with 100 iterations

            p = Process(target=kmm_1.fit, args=(training_set._get_numeric_data(), df_tar._get_numeric_data(), 100, q))
            p.start()
            beta = q.get()
            beta = beta.clip(min=0).flatten()
        
    except MemoryError as err: 
        print("Error: ", err)
        beta = np.array ( [1]*len(training_set) )
        print("Beta is assigned",beta)
        pass

    return beta