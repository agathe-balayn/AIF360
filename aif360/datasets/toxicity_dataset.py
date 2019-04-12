from __future__ import absolute_import

import os

import numpy as np

import pandas as pd

from aif360.datasets.subjectivity_dataset import SubjectivityDataset


from nltk.tokenize import sent_tokenize, word_tokenize, regexp, RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer



def default_preprocessing(df):
    return df

default_mappings = {
    'label_maps': [{1.0: 'Toxic', 0.0: 'Non-toxic'}],
}




class ToxicityDataset(SubjectivityDataset):
    """Wikipedia Toxicity Dataset.
    See :file:`aif360/data/raw/toxicity/README.md`.
    """


    def __init__(self, label_name='toxicity', #favorable_classes=[1],
                 protected_attribute_names=['gender', 'english_first_language', 'age_group', 'education', 'rev_id', 'worker_id'],
                 #privileged_classes=[['male'], lambda x: x >= 25],
                 instance_weights_name=None,
                 categorical_features=[],
                 features_to_keep=[], features_to_drop=[],
                 na_values=[], custom_preprocessing=default_preprocessing,
                 metadata=default_mappings,
                 mapping_categorical_protected=(('gender',('female','male', 'other', 'nan')), ('age_group',('Under 18', '18-30', '30-45', '45-60', 'Over 60', 'nan')), ('education',('none', 'hs', 'some', 'bachelors', 'masters', 'professional', 'doctorate', 'nan')))):

        filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'data', 'raw', 'toxicity')

        ### Read the documents    
        try:
            print("Load toxicity dataset")
            annotations = pd.read_csv(filepath + '/toxicity_annotations.tsv',  sep = '\t', index_col=0)
            annotations = annotations.head(int(len(annotations)/20)) ## Reduce the size for testing
            worker_demo = pd.read_csv(filepath + '/toxicity_worker_demographics.tsv', sep='\t')
        except IOError as err:
            print("IOError: {}".format(err))
            print("To use this class, please download the following files from https://figshare.com/articles/Wikipedia_Talk_Labels_Toxicity/4563973:")
            print("\n\ttoxicity_annotated_comments.tsv")
            print("\ttoxicity_annotations.tsv")
            print("\ttoxicity_worker_demographics.tsv")
            print("\nand place them, as-is, in the folder:")
            print("\n\t{}\n".format(os.path.abspath(os.path.join(
                os.path.abspath(__file__), '..', '..', 'data', 'raw', 'toxicity'))))
            import sys
            sys.exit(1)

        ### Data preparation
        # Data samples preprocessing       
        print("Merge the different datasets") 
        annotations = self.clean_data(annotations, worker_demo, mapping_categorical_protected)
        
        protected_attribute_names = protected_attribute_names + ['pop_label']
        # Data labels and properties computation
        super(ToxicityDataset, self).__init__(df=annotations, label_name=label_name,
            protected_attribute_names=protected_attribute_names,
            instance_weights_name=instance_weights_name,
            categorical_features=categorical_features,
            features_to_keep=features_to_keep,
            features_to_drop=features_to_drop, na_values=na_values,
            custom_preprocessing=custom_preprocessing, metadata=metadata)


    
    def clean_data(self, annotations, worker_demo, mapping_categorical_protected):
        
        # Preprocess workers
        worker_demo = worker_demo.replace(np.NaN, 'nan')
 
         
        #### Add all the information to the annotations
        # Add the worker demographics
        annotations = annotations.reset_index().merge(worker_demo, on='worker_id', how='left').set_index(annotations.index.names)
        # Remove the unknown demographics and the demographics with a NaN. And put them in a general test set.
        annotations = annotations.replace(np.NaN, 'nan')
        annotations.loc[((annotations['english_first_language'].str.contains('nan')) |(annotations['gender'].str.contains('nan')) | (annotations['age_group'].str.contains('nan')) | (annotations['education'].str.contains('nan')) ),'general_split'] = 1 #'test'
        annotations = annotations.reset_index()
        annotations['english_first_language'] = annotations['english_first_language'].replace('nan', 2)
         # MAke the categorical data numbers
        if mapping_categorical_protected != ():
            for tuple_type in mapping_categorical_protected:
                for tuple_details in tuple_type:
                    if tuple_type.index(tuple_details) == 0:
                        key = tuple_details
                    else:
                        for tuple_categories in tuple_details:
                            annotations[key] = annotations[key].replace(tuple_categories, tuple_details.index(tuple_categories))

        annotations['pop_label'] = annotations[['gender', 'age_group', 'education']].apply(lambda x: int(''.join([str(x['gender']),str(x['age_group']), str(x['education'])])), axis=1)    
        if 'general_split' in annotations.columns.tolist():
            annotations = annotations.drop(columns=['general_split'])      

        return annotations
