from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from aif360.datasets import SubjectivityDataset
from aif360.metrics import DatasetMetric, utils
from aif360.metrics import Metric

import pandas as pd
import sklearn.metrics as sk_met
import matplotlib.pyplot as plt


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

def individual_fairness_values(data, metric):
	matrix_data = data.as_matrix([metric])
	# Fairness, general performance
	return 1 - np.nanstd(matrix_data), np.nanmean(matrix_data)

def compute_evaluation_metric(data, GT_col, prediction_col, eval_metrics):
	dict_results = {}
	for metric in eval_metrics:
		if metric == 'F1':
			metric_value = sk_met.f1_score(data[GT_col], data[prediction_col])
		elif metric == 'accuracy':
			metric_value = sk_met.accuracy_score(data[GT_col], data[prediction_col])
		elif metric == 'TPR': # True Positive Rate
			GT_positive_data = data[data[GT_col] == 1]
			if len(GT_positive_data) != 0:
				metric_value = len(GT_positive_data[GT_positive_data[prediction_col] == 1]) / len(GT_positive_data)
			else:
				metric_value = 'NA'
		elif metric == 'TNR': # True Negative Rate
			GT_negative_data = data[data[GT_col] == 0]
			if len(GT_negative_data) != 0:
				metric_value = len(GT_negative_data[GT_negative_data[prediction_col] == 0]) / len(GT_negative_data)
			else:
				metric_value = 'NA' 
		elif metric == 'FPR': # False Positive Rate
			GT_negative_data = data[data[GT_col] == 0]
			if len(GT_negative_data) != 0:
				metric_value = len(GT_negative_data[GT_negative_data[prediction_col] == 1]) / len(GT_negative_data)
			else:
				metric_value = 'NA'
		elif metric == 'FNR': # False Negative Rate
			GT_positive_data = data[data[GT_col] == 1]
			if len(GT_positive_data) != 0:
				metric_value = len(GT_positive_data[GT_positive_data[prediction_col] == 0]) / len(GT_positive_data)
			else:
				metric_value = 'NA'
		dict_results.update({metric:metric_value})	        
	return pd.Series(dict_results)


def annotation_plot_color(value):
	if value > 0.5:
	    return 'b'
	else:
	    return 'w'

def plot_fairness(data, metric, bin_name=''):
	yticks = list(data.index.values)

	# Plot the bins
	data = data.as_matrix(columns=[metric])
	fig, ax = plt.subplots()
	heatmap = ax.pcolor(data)
	cbar = fig.colorbar(heatmap, ax=ax)
	cbar.set_label(metric, rotation=90)
	
	ax.set_yticks(np.arange(data.shape[0]) + 0.5, minor=False)
	ax.set_yticklabels(yticks,rotation=0)
	ax.set_ylabel(bin_name)
	ax.tick_params(left='off', bottom='off',labelbottom='off', color='grey',labelsize='small')

	# Add the exact evaluation measure per bin
	for i in range(data.shape[0]):
	    text = ax.text(0.5, i+0.5, np.round(data[i][0], 2), ha="center", va="center", color=annotation_plot_color(data[i][0]))

	fig.set_figwidth(4)	


def define_fairness_bins(filtering_value, data_filtering_col, comparison_op, data, number_bins):
	if filtering_value is not None:  
		data = data[ops[comparison_op](data[data_filtering_col], filtering_value)]
			
	# Create the bins
	if number_bins is None:
		number_bins = 5 # default value
	data['bin_col'], bin_col = pd.cut(data[data_filtering_col], number_bins, retbins=True)
			
	return data, bin_col


def compute_bin_values(data, binning_entity, GT_col, prediction_col, eval_metrics):
	bin_values = data.drop_duplicates(subset=binning_entity).reset_index()
	bin_values = bin_values.join(bin_values.apply(lambda row: compute_evaluation_metric(data.loc[data[binning_entity] == row[binning_entity]][[GT_col, prediction_col]], GT_col, prediction_col, eval_metrics), axis = 1))
	return bin_values

def compute_bin_fairness(data, fairness_criteria, GT_col, prediction_col, eval_metrics, bin_col, binning_entity):
	# Compute metric per bin
	data = compute_bin_values(data, binning_entity, GT_col, prediction_col, eval_metrics)

	# Aggregate to fairness bins if necessary.
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

	print("TODO: check whether bins are in order")
	return data

class InclusivenessLabelDatasetMetric(Metric):
	"""Class for computing metrics based on a single
	:obj:`~aif360.datasets.SubjectivityLabelDataset`.
	"""

	def __init__(self, dataset):
		"""
		Args:
			dataset (BinaryLabelDataset): A SubjectivityDataset.

		Raises:
			TypeError: `dataset` must be a
				:obj:`~aif360.datasets.SubjectivityDataset` type.
		"""
		if not isinstance(dataset, SubjectivityDataset):
			raise TypeError("'dataset' should be a SubjectivityDataset")

		super(InclusivenessLabelDatasetMetric, self).__init__(dataset)


	
	def compute_metric(self, prediction_col, GT_col, fairness_criteria, eval_metrics, number_bins=None, filtering_value=None, characterization=False):

		# Check whether the predictions of the model are contained in the dataset.  
		if (prediction_col not in self.dataset.dataset.columns) or (GT_col not in self.dataset.dataset.columns):
			raise KeyError("'dataset' should contain the predictions of the model and the ground truth data.")

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

		data = self.dataset.dataset

		# Prepare the bins
		if fairness_criteria != 'demography':
			data, bin_col = define_fairness_bins(filtering_value, data_filtering_col, comparison_op, data, number_bins)
			
		else: # Demography: the bins correspond to the different types of annotators.
			# Remove the lowest frequency demographics categories
			if filtering_value is not None:  
				data = data.groupby(data_filtering_col).filter(lambda x: len(x) >= filtering_value)
			bin_col = 'NA'
		
		# Compute the bin values	
		data = compute_bin_fairness(data, fairness_criteria, GT_col, prediction_col, eval_metrics, bin_col, binning_entity)
		
		# Plot the bins
		if characterization: # Output the characterizations plots per metric
			for metric in eval_metrics: 
				# Plot the distribution
				plot_fairness(data, metric, bin_name=fairness_criteria)

		# Compute the fairness values from the bins
		dict_results = {}
		for metric in eval_metrics: 
			dict_results.update({metric:individual_fairness_values(data, metric)})	        
	
		return dict_results, data