from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from aif360.datasets import SubjectivityDataset
from aif360.metrics import DatasetMetric, utils
from aif360.metrics import Metric

class InclusivenessLabelDatasetMetric(Metric):
	"""Class for computing metrics based on a single
	:obj:`~aif360.datasets.SubjectivityLabelDataset`.
	"""

	def __init__(self, dataset):
		"""
		Args:
			dataset (BinaryLabelDataset): A BinaryLabelDataset.

		Raises:
			TypeError: `dataset` must be a
				:obj:`~aif360.datasets.BinaryLabelDataset` type.
		"""
		if not isinstance(dataset, SubjectivityDataset):
			raise TypeError("'dataset' should be a BinaryLabelDataset")

		super(InclusivenessLabelDatasetMetric, self).__init__(dataset)



	def compute_evaluation_metric(self, data, GT_col, prediction_col, eval_metrics):
		dict_results = {}
		for metric in eval_metrics:
			if metric == 'F1':
				metric_value = sk_met.f1_score(data[data_GT], data[data_prediction])
			elif metric == 'accuracy':
				metric_value = sk_met.accuracy_score(data[data_GT], data[data_prediction])
			elif metric == 'TPR': # True Positive Rate
				GT_positive_data = data[data[GT_col] == 1]
				metric_value = len(GT_positive_data[GT_positive_data[prediction_col] == 1]) / len(GT_positive_data)
			elif metric == 'TNR': # True Negative Rate
				GT_negative_data = data[data[GT_col] == 0]
				metric_value = len(GT_negative_data[GT_negative_data[prediction_col] == 0]) / len(GT_negative_data) 
			elif metric == 'FPR': # False Positive Rate
				GT_negative_data = data[data[GT_col] == 0]
				metric_value = len(GT_negative_data[GT_negative_data[data_prediction] == 1]) / len(GT_negative_data)
			elif metric == 'FNR': # False Negative Rate
				GT_positive_data = data[data[GT_col] == 1]
				metric_value = len(GT_positive_data[GT_positive_data[prediction_col] == 0]) / len(GT_positive_data)
			dict_results.update({metric:metric_value})	        
		return pd.Series(dict_results)

	def compute_metric(self, prediction_col, GT_col, fairness_criteria, eval_metrics, number_bins=None, filtering_value=None, characterization=False):
		#def fairness_metric_3(nb_bin,bin_criteria, bin_min, type_test_data, saving_folder, exp_name, remove_NaN, algo_name, dataset_distribution, training_agg, feature_type):
		
		# Check whether the predictions of the model are contained in the dataset.  
		if (prediction_col not in dataset.columns) or (GT_col not in dataset.columns):
			raise KeyError("'dataset' should contain the predictions of the model and the ground truth data.")

		if fairness_criteria == 'popularity_percentage':
			data_filtering_col = 'popularity_percentage'
			comparison_op = ">="
		elif fairness_criteria == 'annotator_common_perception':
			data_filtering_col = 'annotator_ADR'
			comparison_op = "<="	        
		elif fairness_criteria == 'sample_ambiguity':
			data_filtering_col = 'sample_agreement' 
			comparison_op = ">="	        
		elif bin_criteria == 'demography':
			data_filtering_col = 'pop_label'
			comparison_op = ">="

		
		if fairness_criteria == 'popularity_percentage':
			# Take out the bin ranges which are not studied  
			if filtering_value is not None:  
				data = data[data[data_filtering_col] >= filtering_value]
			
			# Create the bins
			if number_bins is None:
				number_bins = 5 # default value
			data['bin_col'], bin_col = pd.cut(data[data_filtering_col], number_bins, retbins=True)
			
			# Compute the metrics over the bins
			unique_popularity = data.drop_duplicates(subset='bin_col').reset_index()
			
			unique_popularity = unique_popularity.join(unique_popularity.apply(lambda row: compute_evaluation_metric(data.loc[data['bin_col'] == row['bin_col']][[GT_col, prediction_col]], GT_col, prediction_col, eval_metrics), axis = 1))
			data = unique_popularity
			data = data.set_index('bin_col')
			data = data.loc[pd.IntervalIndex.from_breaks(np.round(bin_col,3).astype(np.float64)).values.tolist()]
		elif fairness_criteria == 'demography':
			# Remove the lowest frequency demographics categories
			if filtering_value is not None:  
				data = data.groupby(data_filtering_col).filter(lambda x: len(x) >= filtering_value)
			# Create the bins
			unique_demography = data.drop_duplicates(subset='pop_label').reset_index()
			unique_demography = unique_demography.join(unique_demography.apply(lambda row: compute_evaluation_metric(data.loc[data['pop_label'] == row['pop_label']][[GT_col, prediction_col]], GT_col, prediction_col, eval_metrics), axis = 1))
			data = unique_popularity
			data = data.set_index('bin_col')
			data = data.groupby(data_filtering_col).mean().loc[list_pop_label]
			
		else:
			if number_bins is None:
				number_bins = 5 # default value
			# Take out the bin ranges which are not studied    
			data = data[ops[comparison_op](data[data_filtering_col], filtering_value)]
			# Create the bins
			data['bin_col'], bin_col = pd.cut(data[data_filtering_col], number_bins, retbins=True)

			# Compute the performance per user or per sample
			if fairness_criteria == 'sample_ambiguity':
				unique_samples = data.drop_duplicates(subset='rev_id').reset_index()
				unique_samples = unique_samples.join(unique_samples.apply(lambda row: compute_evaluation_metric(data.loc[data['rev_id'] == row['rev_id']][[GT_col, prediction_col]], GT_col, prediction_col, eval_metrics), axis = 1))
				data = unique_samples
			elif fairness_criteria == 'annotator_common_perception':
				unique_annotators = data.drop_duplicates(subset='worker_id').reset_index()
				unique_annotators = unique_annotators.join(unique_annotators.apply(lambda row: compute_evaluation_metric(data.loc[data['worker_id'] == row['worker_id']][[GT_col, prediction_col]], GT_col, prediction_col, eval_metrics), axis = 1))
				data = unique_annotators
			
			data = data.groupby('bin_col').mean().loc[pd.IntervalIndex.from_breaks(np.round(bin_col,3).astype(np.float64)).values.tolist()]
		
		if characterization: # Output the characterizations per metric
			print("TODO")


		dict_results = {}
		for metric in eval_metrics: 
			dict_results.update({metric:individual_fairness_values(data, metric)})	        
			#pd.Series(dict_results)
	
		return dict_results 

	def individual_fairness_values(self, data, metric):
		matrix_data = data.as_matrix(metric)
		return 1- np.nanstd(matrix_data), np.nanmean(matrix_data)

	def compute_full_fairness_report():
		return 