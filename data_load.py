#############################################################
#                                                           #
#     data_load.py -                                        #
#        Functions for loading relevant datasets            #
#                                                           #
#     Written by Matthew Sheldon & Stefan Tionanda          #
#                                                           #
#############################################################

import pandas as pd
import numpy  as np

import json

DATA_FOLDER = "Data/"

class CommunityViolations:
    """Class wrapper for the "Community Violiations" dataset
    """
    def __init__(self):
        with open (DATA_FOLDER+"com_viol_features.json") as features_file:
            self.feature_names = json.load(features_file)
        
        self.features_to_drop = ["racepctblack","racePctWhite","racePctAsian","racePctHisp",
                                "NumImmig","PctImmigRecent","PctImmigRec5","PctImmigRec8",
                                "PctImmigRec10","PctRecentImmig","PctRecImmig5",
                                "PctRecImmig8","PctRecImmig10","PctSpeakEnglOnly",
                                "PctNotSpeakEnglWell","PctForeignBorn","RacialMatchCommPol",
                                "PctPolicWhite","PctPolicBlack","PctPolicHisp","PctPolicAsian",
                                "PctPolicMinor","AsianPerCap","OtherPerCap", "HispPerCap",
                                "whitePerCap","blackPerCap","indianPerCap","rapes","rapesPerPop"]
        
        self.all_dependent_features = ["murders","murdPerPop","robberies","robbbPerPop", 
                                       "assaults","assaultPerPop","burglaries","burglPerPop",
                                       "larcenies","larcPerPop","autoTheft","autoTheftPerPop",
                                       "ViolentCrimesPerPop","nonViolPerPop","arsons", 
                                       "arsonsPerPop"]
        self.qualitative_features = ['communityCode','communityname', 'state', 'fold', 'population', 'countyCode']
        self.dep_to_drop = ["ViolentCrimesPerPop"] # selected for large number of null values

        self.independent_features = [feat for feat in self.feature_names
                                            if (feat not in self.features_to_drop and \
                                                feat not in self.all_dependent_features and \
                                                feat not in self.qualitative_features)]
        # only select dependent features that end in "PerPop"
        self.dependent_features = [c for c in self.all_dependent_features if c[-6:] == "PerPop"]
        # remove any other specified dependent features
        self.dependent_features = [c for c in self.dependent_features if c not in self.dep_to_drop]
        self.data_cache = None
        self.file_name = 'CommViolPredUnnormalizedData.txt'
        self.file_loc = DATA_FOLDER + self.file_name
    
    def load_data_from_souce(self):
        """Load directly from the file. Calls self.clean_data as well."
        """
        unclean_data = pd.read_csv(self.file_loc, names = self.feature_names)
        self.data_cache = self.clean_data(unclean_data)

    def clean_data(self, unclean_data):
        """Takes in an unclean, raw version of the dataset and applies the following:
           - Remove any undesired columns
           - Split the dataset into independent and dependent features
           - Remove any rows with a single null
        """
        null_entries = unclean_data[self.independent_features + self.dependent_features] == "?"
        null_entries = null_entries.any(axis=1)
        nnull_entries = ~null_entries
        independent_data = unclean_data[nnull_entries][self.independent_features]
        dependent_data   = unclean_data[nnull_entries][self.dependent_features]

        return [independent_data, dependent_data]

    def pull_data(self):
        """Returns a pandas dataframe of the desired dataset. 
                If the data cache is none, it loads and filters as necessary.
        """
        if not self.data_cache:
            self.load_data_from_souce()
        return self.data_cache