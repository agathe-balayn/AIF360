import numpy as np
import pandas as pd

from aif360.datasets import Dataset

def rescore_func(x):
    if x < 0.5:
        return 1-x
    else:
        return x

class SubjectivityDataset(Dataset):
    """Base class for all datasets of subjective properties.
    A SubjectivityDataset requires data to be stored in :obj:`numpy.ndarray`
    objects with :obj:`~numpy.dtype` as :obj:`~numpy.float64`.
    Attributes:
        TO FILL IN
    """

    def __init__(self, samples, annotations, custom_preprocessing=None, metadata=None):
        """
        Args:
            samples: dataframe of samples
            annotations: dataframe of corresponding annotations
            metadata (optional): Additional metadata to append.
        Raises:
            TypeError: Certain fields must be np.ndarrays as specified in the
                class description.
            ValueError: ndarray shapes must match.
        """

        """
        if df is None:
            raise TypeError("Must provide a pandas DataFrame representing "
                            "the data (features, labels, protected attributes)")
        if df.isna().any().any():
            raise ValueError("Input DataFrames cannot contain NA values.")
        try:
            df = df.astype(np.float64)
        except ValueError as e:
            print("ValueError: {}".format(e))
            raise ValueError("DataFrame values must be numerical.")
        """

        # Convert all column names to strings
        #df.columns = df.columns.astype(str).tolist()
        #label_names = list(map(str, label_names))
        #protected_attribute_names = list(map(str, protected_attribute_names))

        self.samples = samples
        self.annotations = annotations

        
        # always ignore metadata and ignore_fields
        self.ignore_fields = {'metadata', 'ignore_fields'}

        # Compute the different scores later used for computing the fairness metrics.
        self.samples, self.annotations = self.compute_scores(self.samples, self.annotations)

        # Merge the data into a unique dataset structure.
        self.dataset = self.annotations.merge(self.samples[['rev_id', 'comment']], on='rev_id', how='left')

        # sets metadata
        super(SubjectivityDataset, self).__init__(df=self.dataset, metadata=metadata)

    
    def compute_scores(self, comments, annotations):
        ### Compute the different scores exploited by the fairness metrics.
        ## Compute the comments' majority vote, average toxicity score and ambiguity.
        ## Compute the annotator average disagreement with the majority vote.
        ## Compute the annotations' popularity among the other annotations.
        
        comments = comments.reset_index().merge(annotations.groupby('rev_id')['label'].mean().to_frame('average_label').reset_index(), on='rev_id', how='left')
        
        # Get the comments' ambiguity (percentage of agreement in the annotations)
        comments['sample_agreement'] = comments['average_label'].apply(lambda x: rescore_func(x))
        # Propagate to the annotations
        annotations = annotations.merge(comments[['rev_id', 'sample_agreement']], on='rev_id', how='left')

        # Get the majority vote associated to each sample
        annotations['MV'] = annotations.groupby(['rev_id'])['label'].transform(lambda x : (x.mean() >= 0.5).astype(int))

        # Get the annotators' average disagreement rate (ADR) with the majority vote
        #annotations['average_label'] = annotations.groupby(['rev_id'])['label'].transform(lambda x : x.mean())
        annotations['disaggreement_bin'] = 1 - (annotations['label'] == annotations['MV']).astype(int)
        annotations['annotator_ADR'] = annotations.groupby('worker_id')['disaggreement_bin'].transform(lambda x: x.mean())#.reset_index()
        annotations = annotations.drop(columns=['disaggreement_bin'])

        # Annotation popularity
        annotations['popularity_number'] = annotations.groupby(['rev_id', 'label'])['rev_id'].transform(lambda x: len(x))
        annotations['popularity_percentage'] = annotations.groupby(['rev_id'])['popularity_number'].transform(lambda x: x/len(x))#x['popularity_percentage']/len(x))
        annotations = annotations.drop(columns=['popularity_number'])

        
        return comments, annotations


    def export_dataset(self, export_metadata=False):
        """
        Export the dataset and supporting attributes
        TODO: The preferred file format is HDF
        """

        if export_metadata:
            raise NotImplementedError("The option to export metadata has not been implemented yet")

        return None

    def import_dataset(self, import_metadata=False):
        """ Import the dataset and supporting attributes
            TODO: The preferred file format is HDF
        """

        if import_metadata:
            raise NotImplementedError("The option to import metadata has not been implemented yet")
        return None

    def split(self, num_or_size_splits, shuffle=False, seed=None):
        """Split the dataset into multiple datasets
        Args:
            num_or_size_splits (list or int):
            shuffle (bool):
            seed (int or array_like): takes the same argument as `numpy.random.seed()`
            function
        Returns:
            list: Each element of this list is a dataset obtained during the split
        """

        # Set seed
        if seed is not None:
            np.random.seed(seed)


        # Only perform the split on the annotations whose annotators information are known.
        dataset_split = self.dataset[self.dataset['general_split'] == 'train'].copy().reset_index()

        n = dataset_split.shape[0]
        if isinstance(num_or_size_splits, list):
            num_folds = len(num_or_size_splits) + 1
            if num_folds > 1 and all(x <= 1. for x in num_or_size_splits):
                num_or_size_splits = [int(x * n) for x in num_or_size_splits]
        else:
            num_folds = num_or_size_splits

        order = list(np.random.permutation(n) if shuffle else range(n))
        folds = [self.copy() for _ in range(num_folds)]

        dataset = np.array_split(dataset_split.iloc[order, :], num_or_size_splits)
        
        for fold, data in zip(
                folds, dataset):

            fold.dataset = data
            fold.metadata = fold.metadata.copy()
            fold.metadata.update({
                'transformer': '{}.split'.format(type(self).__name__),
                'params': {'num_or_size_splits': num_or_size_splits,
                           'shuffle': shuffle},
                'previous': [self]
            })

        return folds

    @staticmethod
    def _de_dummy_code_df(df, sep="=", set_category=False):
        """De-dummy code a dummy-coded dataframe obtained with pd.get_dummies().
        After reversing dummy coding the corresponding fields will be converted
        to categorical.
        Args:
            df (pandas.DataFrame): Input dummy coded dataframe
            sep (char): Separator between base name and dummy code
            set_category (bool): Set the de-dummy coded features
                    to categorical type
        Examples:
            >>> columns = ["Age", "Gender=Male", "Gender=Female"]
            >>> df = pd.DataFrame([[10, 1, 0], [20, 0, 1]], columns=columns)
            >>> _de_dummy_code_df(df, sep="=")
               Age  Gender
            0   10    Male
            1   20  Female
        """

        feature_names_dum_d, feature_names_nodum = \
            StructuredDataset._parse_feature_names(df.columns)
        df_new = pd.DataFrame(index=df.index,
            columns=feature_names_nodum + list(feature_names_dum_d.keys()))

        for fname in feature_names_nodum:
            df_new[fname] = df[fname].values.copy()

        for fname, vl in feature_names_dum_d.items():
            for v in vl:
                df_new.loc[df[fname+sep+str(v)] == 1, fname] = str(v)

        if set_category:
            for fname in feature_names_dum_d.keys():
                df_new[fname] = df_new[fname].astype('category')

        return df_new

    @staticmethod
    def _parse_feature_names(feature_names, sep="="):
        """Parse feature names to ordinary and dummy coded candidates.
        Args:
            feature_names (list): Names of features
            sep (char): Separator to designate the dummy coded category in the
                feature name
        Returns:
            (dict, list):
                * feature_names_dum_d (dict): Keys are the base feature names
                  and values are the categories.
                * feature_names_nodum (list): Non-dummy coded feature names.
        Examples:
            >>> feature_names = ["Age", "Gender=Male", "Gender=Female"]
            >>> StructuredDataset._parse_feature_names(feature_names, sep="=")
            (defaultdict(<type 'list'>, {'Gender': ['Male', 'Female']}), ['Age'])
        """
        feature_names_dum_d = defaultdict(list)
        feature_names_nodum = list()
        for fname in feature_names:
            if sep in fname:
                fname_dum, v = fname.split(sep, 1)
                feature_names_dum_d[fname_dum].append(v)
            else:
                feature_names_nodum.append(fname)

        return feature_names_dum_d, feature_names_nodum