from __future__ import absolute_import

import os

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

    def __init__(self, custom_preprocessing=default_preprocessing,
                 metadata=default_mappings):

        filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'data', 'raw', 'toxicity')

        ### Read the documents    
        try:
            comments = pd.read_csv(filepath + '/toxicity_annotated_comments.tsv', sep = '\t', dtype={'rev_id':int, 'comment':str}, index_col = 0)
            annotations = pd.read_csv(filepath + '/toxicity_annotations.tsv',  sep = '\t', index_col=0)
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


        ### Merge the tables


        ### Clean the data
        # Data preparing
        
        comments, annotations, worker_demo = self.clean_data(comments, annotations, worker_demo)
        super(ToxicityDataset, self).__init__(samples=comments, annotations=annotations,
            custom_preprocessing=custom_preprocessing, 
            metadata=metadata)



    def normalize_text(self, text):
        tokenizer = RegexpTokenizer(r'\w+')
        stopword_set = set(stopwords.words('english'))
        stemmer = PorterStemmer()
        # Convert text to lower-case and strip punctuation/symbols from words
        norm_text = text.lower()
        # Replace breaks with spaces
        norm_text = norm_text.replace('<br />', ' ')
        # Pad punctuation with spaces on both sides
        for char in ['.', '"', ',', '(', ')', '!', '?', ';', ':']:
            norm_text = norm_text.replace(char, ' ' + char + ' ') 
        # Tokenize
        norm_text = tokenizer.tokenize(norm_text)
        # Remove stop words
        norm_text = [w for w in norm_text if not w in stopword_set]
        #norm_text = list(set(norm_text).difference(stopword_set))
        norm_text = " ".join(norm_text)
        #print(norm_text)
        # Stem words
        #norm_text = [stemmer.stem(i) for i in norm_text]
        # remove empty
        #norm_text = [i for i in norm_text if len(i) > 1]
        return norm_text

    
    def clean_data(self, comments, annotations, worker_demo):
        comments['comment'] = comments['comment'].apply(lambda x: x.replace("NEWLINE_TOKEN", " "))
        comments['comment'] = comments['comment'].apply(lambda x: x.replace("TAB_TOKEN", " "))
        comments['comment'] = comments['comment'].apply(lambda x: self.normalize_text(x))
        comments = comments.drop('year', 1)
        comments = comments.drop('logged_in', 1)
        comments = comments.drop('ns', 1)
        comments = comments.drop('sample', 1)
        comments = comments.drop('split', 1)
        # Preprocess workers
        worker_demo['pop_label'] = worker_demo[['gender', 'age_group', 'education']].apply(lambda x: ' '.join([str(x['gender']),str(x['age_group']), str(x['education'])]), axis=1)
        # Clean annotations
        #annotations = annotations.drop('comment',1)
        #annotations = annotations.drop('year',1)
        #annotations = annotations.drop('logged_in',1)
        #annotations = annotations.drop('ns',1)
        #annotations = annotations.drop('sample',1)
        #annotations = annotations.drop('split',1)
        #annotations = annotations.drop('toxicity_label_0',1)
        
        #### Add all the information to the annotations
        ## Add the worker demographics
        annotations = annotations.reset_index().merge(worker_demo, on='worker_id', how='left').set_index(annotations.index.names)
        ### Remove the unknown demographics and the demographics with a NaN. And put them ina general test set.
        annotations['general_split'] = 'train'
        annotations.loc[((annotations['pop_label'].isnull()) | (annotations['pop_label'].str.contains('nan')) ),'general_split'] = 'test'
        return comments, annotations, worker_demo

