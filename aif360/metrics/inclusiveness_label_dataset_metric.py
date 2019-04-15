from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import copy

from aif360.datasets import SubjectivityDataset, BinaryLabelDataset
from aif360.metrics import ClassificationMetric, utils
from aif360.metrics import Metric

import pandas as pd
import sklearn.metrics as sk_met
import matplotlib.pyplot as plt

import os

import operator

ops = {
"==": operator.eq,
"!=": operator.ne,
"<>": operator.ne,
"<": operator.lt,
"<=": operator.le,
">": operator.gt,
">=": operator.ge
}



class InclusivenessLabelDatasetMetric(ClassificationMetric):
	"""Class for computing metrics based on a single
	:obj:`~aif360.datasets.SubjectivityLabelDataset`.
	"""

	def __init__(self, dataset, classified_dataset,
                 unprivileged_groups=None, privileged_groups=None):
		"""
		Args:
			dataset (BinaryLabelDataset): A SubjectivityDataset.

		Raises:
			TypeError: `dataset` must be a
				:obj:`~aif360.datasets.SubjectivityDataset` type.
		"""

		self.dataset = dataset
		if not isinstance(dataset, SubjectivityDataset):
			raise TypeError("'dataset' should be a SubjectivityDataset")

		if isinstance(classified_dataset, BinaryLabelDataset):
			self.classified_dataset = classified_dataset
		else:
			raise TypeError("'classified_dataset' should be a "
                            "BinaryLabelDataset.")
		
		super(InclusivenessLabelDatasetMetric, self).__init__(dataset, classified_dataset,
                 unprivileged_groups=None, privileged_groups=None)

	def get_index_bin_metric(self, fairness_criteria, number_bins=None, filtering_value=None):
		# Type of fairness
		if fairness_criteria == 'annotation_popularity':
			data_filtering_col = 'popularity_percentage'
			binning_entity = 'bin_col'
			comparison_op = ">="
		elif fairness_criteria == 'annotator_disagreement':
			data_filtering_col = 'annotator_ADR'
			comparison_op = "<="	        
			binning_entity = 'worker_id'
		elif fairness_criteria == 'sample_ambiguity':
			data_filtering_col = 'sample_agreement' 
			binning_entity = 'rev_id'
			comparison_op = ">="	        
		elif fairness_criteria == 'demography':
			data_filtering_col = 'pop_label'
			binning_entity = 'pop_label'
			comparison_op = ">="

		data = pd.DataFrame(data=self.dataset.features, columns=self.dataset.feature_names)

		# Prepare the bins
		if fairness_criteria != 'demography':
			data, bin_col = define_fairness_bins(filtering_value, data_filtering_col, comparison_op, data, number_bins)
			
		else: # Demography: the bins correspond to the different types of annotators.
			# Remove the lowest frequency demographics categories.
			if filtering_value is not None:  
				data = data.groupby(data_filtering_col).filter(lambda x: len(x) >= filtering_value)
			bin_col = 'NA'

		return data, bin_col, binning_entity
	

	

	def compute_bin_metrics(self, fairness_criteria, eval_metrics=('accuracy'), number_bins=None, filtering_value=None):
		eval_metrics = list(eval_metrics)
		data, bin_col, binning_entity = self.get_index_bin_metric(fairness_criteria, number_bins, filtering_value)
		
		# Compute the bin values	
		### Compute metric per bin (individual users, comments, ...)
		bin_values = data.drop_duplicates(subset=binning_entity)
		
		# Create the list of the results.
		list_bin_metric = []
		for i in range(0,len(eval_metrics)):
			list_bin_metric.append([])
		# Create a new BinaryLabelDataset per bin and get the values 
		for i, j in bin_values.iterrows():
			# Get the index
			list_index = ((data.index[data[binning_entity] == j[binning_entity]]).tolist())
			# Create the new classes
			bin_metric_class = get_bin_data(self.dataset, self.classified_dataset, list_index)
			# Compute the metrics and save the metric values for each bin
			for metric_to_compute in eval_metrics:
				if metric_to_compute == 'accuracy':
					list_bin_metric[eval_metrics.index(metric_to_compute)].append(bin_metric_class.accuracy())
				elif metric_to_compute == 'TPR':
					list_bin_metric[eval_metrics.index(metric_to_compute)].append(bin_metric_class.true_positive_rate())
				elif metric_to_compute == 'TNR':
					list_bin_metric[eval_metrics.index(metric_to_compute)].append(bin_metric_class.true_negative_rate())
				elif metric_to_compute == 'FPR':
					list_bin_metric[eval_metrics.index(metric_to_compute)].append(bin_metric_class.false_positive_rate())
				elif metric_to_compute == 'FNR':
					list_bin_metric[eval_metrics.index(metric_to_compute)].append(bin_metric_class.false_negative_rate())

		### Get the average over the larger bins
		for metric_to_compute in eval_metrics:
			bin_values[metric_to_compute] = list_bin_metric[eval_metrics.index(metric_to_compute)]

		data = bin_values
		if fairness_criteria == 'annotation_popularity':
			data = data.set_index('bin_col')
		elif fairness_criteria == 'demography':
			data = data.set_index('pop_label')
			list_pop_label = list(data.index.values)
			data = data.groupby('pop_label').mean()
		else:			
			data = data.groupby('bin_col').mean()

		# Reorder the bins
		if fairness_criteria == 'demography':
			data = data.loc[list_pop_label]
		else:
			data = data.loc[pd.IntervalIndex.from_breaks(np.round(bin_col,3).astype(np.float64)).values.tolist()]	

		results_to_return = []
		for metric_to_compute in eval_metrics:
			matrix_data = data.as_matrix([metric_to_compute])
			# Fairness, general performance, the performance for each bin
			results_to_return.append((1 - np.nanstd(matrix_data), np.nanmean(matrix_data), data[[metric_to_compute]]))
		return results_to_return

def get_subset_BinaryLabelDataset(dataset, original_dataset, list_index):
	dataset.feature_names = original_dataset.feature_names
	dataset.features = original_dataset.features[np.ix_(list_index)]

	dataset.label_names = original_dataset.label_names
	dataset.labels = original_dataset.labels[np.ix_(list_index)]

	dataset.scores = original_dataset.scores[np.ix_(list_index)]

	dataset.instance_names = original_dataset.instance_names
	dataset.instance_weights = original_dataset.instance_weights[np.ix_(list_index)]

	dataset.protected_attribute_names = original_dataset.protected_attribute_names
	dataset.protected_attributes = original_dataset.protected_attributes[np.ix_(list_index)]

	dataset.unprivileged_protected_attributes = original_dataset.unprivileged_protected_attributes
	dataset.privileged_protected_attributes = original_dataset.privileged_protected_attributes

	return dataset

def get_bin_data(dataset, classified_dataset, list_index):
	dataset_GT = empty_subjectivity_dataset_wrapper() 
	dataset_GT = get_subset_BinaryLabelDataset(dataset_GT, dataset, list_index)
	dataset_prediction = empty_subjectivity_dataset_wrapper()
	dataset_prediction = get_subset_BinaryLabelDataset(dataset_prediction, classified_dataset, list_index)
	return InclusivenessLabelDatasetMetric(dataset_GT, dataset_prediction)	

def define_fairness_bins(filtering_value, data_filtering_col, comparison_op, data, number_bins):
	if filtering_value is not None:  
		data = data[ops[comparison_op](data[data_filtering_col], filtering_value)]
			
	# Create the bins
	if number_bins is None:
		number_bins = 5 # default value
	data['bin_col'], bin_col = pd.cut(data[data_filtering_col], number_bins, retbins=True)
			
	return data, bin_col



def default_preprocessing(df):
    return df
def empty_subjectivity_dataset_wrapper():
    annotations_empty = pd.DataFrame(data={'rev_id':[0,0], 'worker_id':[0,0], 'toxicity':[0,1], 'toxicity_score':[0.0, 1.0], 'gender':['nan', 'nan'], 'english_first_language':[2, 2], 'age_group':['nan', 'nan'], 'education':['nan', 'nan'], 'pop_label':['1 1 1', '1 1 1'], 'GT':[0, 1], 'MV':[0, 1]})
    mapping_categorical_protected=(('gender',('female','male', 'other', 'nan')), ('age_group',('Under 18', '18-30', '30-45', '45-60', 'Over 60', 'nan')), ('education',('none', 'hs', 'some', 'bachelors', 'masters', 'professional', 'doctorate', 'nan')))
    for tuple_type in mapping_categorical_protected:
        for tuple_details in tuple_type:
            if tuple_type.index(tuple_details) == 0:
                key = tuple_details
            else:
                for tuple_categories in tuple_details:
                    annotations_empty[key] = annotations_empty[key].replace(tuple_categories, tuple_details.index(tuple_categories))
    annotations_empty['pop_label'] = annotations_empty[['gender', 'age_group', 'education']].apply(lambda x: int(''.join([str(x['gender']),str(x['age_group']), str(x['education'])])), axis=1)    

    annotations_empty = annotations_empty[['rev_id', 'worker_id', 'toxicity', 'toxicity_score', 'gender', 'english_first_language', 'age_group', 'education', 'pop_label', 'GT', 'MV']]

    protected_attribute_names=['gender', 'english_first_language', 'age_group', 'education', 'rev_id', 'worker_id', 'pop_label']
    privileged_classes=None
    instance_weights_name=None
    categorical_features=[]
    features_to_keep=[]
    features_to_drop=[]
    na_values=[]
    custom_preprocessing=default_preprocessing
    metadata={'label_maps': [{1.0: 'Toxic', 0.0: 'Non-toxic'}],}
    dataset = SubjectivityDataset(annotations_empty, 'toxicity', 'GT',
                 protected_attribute_names=protected_attribute_names, privileged_classes=privileged_classes,
                 instance_weights_name=instance_weights_name, categorical_features=categorical_features,
                 features_to_keep=features_to_keep, features_to_drop=features_to_drop, na_values=na_values,
                 custom_preprocessing=custom_preprocessing, metadata=metadata)
    return dataset