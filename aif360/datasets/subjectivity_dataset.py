import numpy as np
import pandas as pd
from logging import warn

from aif360.datasets import BinaryLabelDataset

def rescore_func(x):
    if x < 0.5:
        return 1-x
    else:
        return x

def compute_sample_agreement(x, label_name):
    x['sample_agreement'] = rescore_func(x[label_name].mean())
    return x

class SubjectivityDataset(BinaryLabelDataset):
    """Base class for all datasets of subjective properties.
    A SubjectivityDataset requires data to be stored in :obj:`numpy.ndarray`
    objects with :obj:`~numpy.dtype` as :obj:`~numpy.float64`.
    Attributes:
    """


    def __init__(self, df, label_name, #favorable_classes,
                 protected_attribute_names, privileged_classes=None,
                 instance_weights_name='', categorical_features=[],
                 features_to_keep=[], features_to_drop=[], na_values=[],
                 custom_preprocessing=None, metadata=None):

        # 2. Dataset preprocessing
        if custom_preprocessing:
            df = custom_preprocessing(df)

        if label_name != 'labels':
            df.rename(columns={label_name:'labels'}, inplace=True)
            label_name = 'labels'
        # Compute the different scores later used for computing the fairness metrics.
        #print("Compute the scores describing the dataset, used for the evaluation metric.")
        df = self.compute_scores(df, label_name)
        protected_attribute_names = protected_attribute_names + ['sample_agreement', 'annotator_ADR', 'popularity_percentage']
        
        # 3. Drop unrequested columns
        features_to_keep = features_to_keep or df.columns.tolist()
        keep = (set(features_to_keep) | set(protected_attribute_names)
              | set(categorical_features) | set([label_name]))
        if instance_weights_name:
            keep |= set([instance_weights_name])
        df = df[list(keep - set(features_to_drop))]

        # 4. Remove any rows that have missing data.
        dropped = df.dropna()
        count = df.shape[0] - dropped.shape[0]
        if count > 0:
            warn("Missing Data: {} rows removed from {}.".format(count,
                type(self).__name__))
        df = dropped

        # 5. Create a one-hot encoding of the categorical variables.
        df = pd.get_dummies(df, columns=categorical_features, prefix_sep='=')

      
        super(SubjectivityDataset, self).__init__(df=df, label_names=[label_name],
            protected_attribute_names=protected_attribute_names,
            instance_weights_name=instance_weights_name,
            metadata=metadata)



   
    def compute_scores(self, annotations, label_name):
        ### Compute the different scores exploited by the fairness metrics.
        ## Compute the comments' majority vote, average toxicity score and ambiguity.
        ## Compute the annotator average disagreement with the majority vote.
        ## Compute the annotations' popularity among the other annotations.
        annotations = annotations.groupby('rev_id').apply(compute_sample_agreement,label_name=label_name)
        # Get the majority vote associated to each sample
        annotations['MV'] = annotations.groupby(['rev_id'])[label_name].transform(lambda x : (x.mean() >= 0.5).astype(int))

        # Get the annotators' average disagreement rate (ADR) with the majority vote
        #annotations['average_label'] = annotations.groupby(['rev_id'])['label'].transform(lambda x : x.mean())
        annotations['disaggreement_bin'] = 1 - (annotations[label_name] == annotations['MV']).astype(int)
        annotations['annotator_ADR'] = annotations.groupby('worker_id')['disaggreement_bin'].transform(lambda x: x.mean())#.reset_index()
        annotations = annotations.drop(columns=['disaggreement_bin'])

        # Annotation popularity
        annotations['popularity_number'] = annotations.groupby(['rev_id', label_name])['rev_id'].transform(lambda x: len(x))
        annotations['popularity_percentage'] = annotations.groupby(['rev_id'])['popularity_number'].transform(lambda x: x/len(x))#x['popularity_percentage']/len(x))
        annotations = annotations.drop(columns=['popularity_number'])

        
        return annotations

