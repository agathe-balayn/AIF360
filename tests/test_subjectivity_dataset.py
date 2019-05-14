from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import pandas as pd

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 200)


import os
from aif360.datasets import SubjectivityDataset
from aif360.datasets import ToxicityDataset


def test_toxicity():
    # just test that there are no errors for default loading...
    td = ToxicityDataset()

def default_preprocessing(df):
    return df

def test_subjectivity():
    # Construct a random subjectivity dataset.
    
    cols = ['rev_id', 'worker_id', 'toxicity', 'gender', 'english_first_language', 'age_group', 'education', 'pop_label', 'GT'] #,'MV']
    nb_annotations = (100, 1)
    rev_id = np.random.randint(10, size=nb_annotations) #np.arange(nb_annotations)
    worker_id = np.random.randint(20, size=nb_annotations)
    toxicity_labs = np.random.randint(2, size=nb_annotations)
    GT_labs = np.random.randint(2, size=nb_annotations)

    gender_labs = np.random.randint(4, size=nb_annotations)
    english_labs = np.random.randint(2, size=nb_annotations)
    age_labs = np.random.randint(6, size=nb_annotations)
    education_labs = np.random.randint(8, size=nb_annotations)

    df = pd.DataFrame(data=np.concatenate((rev_id, worker_id, toxicity_labs, gender_labs, english_labs, age_labs, education_labs), axis=1), columns=cols[:-2])
    df['pop_label'] = df[['gender', 'age_group', 'education']].apply(lambda x: int(''.join([str(x['gender']),str(x['age_group']), str(x['education'])])), axis=1)    
    df['GT'] = GT_labs

    protected_attribute_names=['gender', 'english_first_language', 'age_group', 'education', 'rev_id', 'worker_id', 'pop_label']
    privileged_classes=None
    instance_weights_name=None
    categorical_features=[]
    features_to_keep=[]
    features_to_drop=[]
    na_values=[]
    custom_preprocessing=default_preprocessing
    metadata={'label_maps': [{1.0: 'Toxic', 0.0: 'Non-toxic'}],}

    sd = SubjectivityDataset(df, 'toxicity', 'GT',
        protected_attribute_names=protected_attribute_names, privileged_classes=privileged_classes,
        instance_weights_name=instance_weights_name, categorical_features=categorical_features,
        features_to_keep=features_to_keep, features_to_drop=features_to_drop, na_values=na_values,
        custom_preprocessing=custom_preprocessing, metadata=metadata)
    
    # Check its properties
    assert sd.labels.shape[0] == nb_annotations[0]
    assert sd.features.shape[0] == nb_annotations[0]
    assert 'sample_agreement' in sd.feature_names
    assert 'popularity_percentage' in sd.feature_names
    assert 'annotator_ADR' in sd.feature_names
    assert 'MV' in sd.feature_names
    assert np.logical_and(sd.features[:,sd.feature_names.index('sample_agreement')] >= 0, sd.features[:,sd.feature_names.index('sample_agreement')] <= 1).all()
    assert np.logical_and(sd.features[:,sd.feature_names.index('popularity_percentage')] >= 0, sd.features[:,sd.feature_names.index('popularity_percentage')] <= 1).all()
    assert np.logical_and(sd.features[:,sd.feature_names.index('annotator_ADR')] >= 0, sd.features[:,sd.feature_names.index('annotator_ADR')] <= 1).all()
