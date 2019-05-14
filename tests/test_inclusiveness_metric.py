from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import pandas as pd

from aif360.datasets import BinaryLabelDataset
from aif360.metrics import InclusivenessLabelDatasetMetric


def test_extreme_cases():
    
    #### Test with ML models with 100% accuracy:
    # First dataset    
    cols = ['rev_id', 'worker_id', 'toxicity', 'gender', 'english_first_language', 'age_group', 'education', 'pop_label', 'GT'] #,'MV']
    nb_annotations = (100, 1)
    rev_id = np.random.randint(10, size=nb_annotations) #np.arange(nb_annotations)
    worker_id = np.random.randint(20, size=nb_annotations)
    toxicity_labs = np.concatenate((np.ones((int(nb_annotations[0]/2), nb_annotations[1]), dtype=int), np.zeros((int(nb_annotations[0]/2), nb_annotations[1]), dtype=int)), axis=0) 
    GT_labs = toxicity_labs

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
    sd1 = SubjectivityDataset(df.copy(), 'toxicity', 'GT',
        protected_attribute_names=protected_attribute_names, privileged_classes=privileged_classes,
        instance_weights_name=instance_weights_name, categorical_features=categorical_features,
        features_to_keep=features_to_keep, features_to_drop=features_to_drop, na_values=na_values,
        custom_preprocessing=custom_preprocessing, metadata=metadata)
    
    # Second dataset
    sd2 = sd1.copy()
    
    test_metric_inclusion = InclusivenessLabelDatasetMetric(sd1, sd2)
    
    res = test_metric_inclusion.average_accuracy('annotator_disagreement', number_bins=None, filtering_value=None)
    assert res[0] == 1
    assert res[1] == 1
    res = test_metric_inclusion.true_positive_rate('annotator_disagreement', number_bins=None, filtering_value=None)
    assert res[0] == 1
    assert res[1] == 1
    res = test_metric_inclusion.average_accuracy('annotation_popularity', number_bins=None, filtering_value=None)
    assert res[0] == 1
    assert res[1] == 1
    res = test_metric_inclusion.average_accuracy('sample_ambiguity', number_bins=None, filtering_value=None)
    assert res[0] == 1
    assert res[1] == 1
    res = test_metric_inclusion.average_accuracy('demography', number_bins=None, filtering_value=None)
    assert res[0] == 1
    assert res[1] == 1
    
    #### Test with ML models with 0% accuracy:
    # Third dataset
    df3 = df.copy()
    df3.loc[:,'toxicity'] = np.concatenate((np.zeros((int(nb_annotations[0]/2), nb_annotations[1]), dtype=int), np.ones((int(nb_annotations[0]/2), nb_annotations[1]), dtype=int)), axis=0) # np.random.randint(2, size=nb_annotations) #####

    sd3 = SubjectivityDataset(df3, 'toxicity', 'GT',
        protected_attribute_names=protected_attribute_names, privileged_classes=privileged_classes,
        instance_weights_name=instance_weights_name, categorical_features=categorical_features,
        features_to_keep=features_to_keep, features_to_drop=features_to_drop, na_values=na_values,
        custom_preprocessing=custom_preprocessing, metadata=metadata)

    test_metric_inclusion2 = InclusivenessLabelDatasetMetric(sd1, sd3)
    
    res = test_metric_inclusion2.average_accuracy('annotator_disagreement', number_bins=None, filtering_value=None)
    assert res[0] == 1
    assert res[1] == 0
    res = test_metric_inclusion2.true_positive_rate('annotator_disagreement', number_bins=None, filtering_value=None)
    assert res[0] == 1
    assert res[1] == 0
    res = test_metric_inclusion2.average_accuracy('annotation_popularity', number_bins=None, filtering_value=None)
    assert res[0] == 1
    assert res[1] == 0
    res = test_metric_inclusion2.average_accuracy('sample_ambiguity', number_bins=None, filtering_value=None)
    assert res[0] == 1
    assert res[1] == 0
    res = test_metric_inclusion2.average_accuracy('demography', number_bins=None, filtering_value=None)
    assert res[0] == 1
    assert res[1] == 0
    
    ### Test with ML model of "random"/unknown accuracy
    # Fourth dataset
    df4 = df.copy()
    df4.loc[:, 'toxicity'] = np.random.randint(2, size=nb_annotations)
    
    sd4 = SubjectivityDataset(df4, 'toxicity', 'GT',
        protected_attribute_names=protected_attribute_names, privileged_classes=privileged_classes,
        instance_weights_name=instance_weights_name, categorical_features=categorical_features,
        features_to_keep=features_to_keep, features_to_drop=features_to_drop, na_values=na_values,
        custom_preprocessing=custom_preprocessing, metadata=metadata)

    test_metric_inclusion3 = InclusivenessLabelDatasetMetric(sd1, sd4)
    
    res = test_metric_inclusion3.average_accuracy('annotator_disagreement', number_bins=None, filtering_value=None)
    assert res[0] < 1
    assert res[1] > 0
    res = test_metric_inclusion3.true_positive_rate('annotator_disagreement', number_bins=None, filtering_value=None)
    assert res[0] < 1
    assert res[1] > 0
    res = test_metric_inclusion3.average_accuracy('annotation_popularity', number_bins=None, filtering_value=None)
    assert res[0] < 1
    assert res[1] > 0
    res = test_metric_inclusion3.average_accuracy('sample_ambiguity', number_bins=None, filtering_value=None)
    assert res[0] < 1
    assert res[1] > 0
    res = test_metric_inclusion3.average_accuracy('demography', number_bins=None, filtering_value=None)
    assert res[0] < 1
    assert res[1] > 0