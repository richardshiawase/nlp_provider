import pandas as pd
from django.core.cache import cache

from classM.Pembersih import Pembersih


class Dataset:
    def __init__(self,pd):
        self.df_dataset = cache.get('dataset')
        if self.df_dataset is None:
            self.df_dataset = pd.read_excel("dataset_excel_copy.xlsx")
            cache.set('dataset', self.df_dataset)

        self.bulk_dataset_copy = self.df_dataset.copy()
        self.bulk_dataset_copy['course_title'] = self.bulk_dataset_copy['course_title'].str.lower()
        # self.df_dataset_no_duplicate = self.bulk_dataset.drop_duplicates(['course_title'],keep='first')
        # self.df_dataset['co'] = self.df_dataset.dropna()
        # self.df_dataset['course_title'] = self.df_dataset.drop_duplicates(['course_title'],keep='first')
        self.pembersih = Pembersih((self.bulk_dataset_copy.drop_duplicates(['course_title'],keep='first')))

    def get_bulk_dataset(self):
        return self.df_dataset

    def set_list_item_dataset(self,list_item_dataset):
        self.list_item_dataset = list_item_dataset

    def get_list_item_dataset(self):
        return self.list_item_dataset

    def set_course_title_list(self,course_title_list):
        self.course_title_list = course_title_list

    def get_dataframe_after_cleaned_no_duplicate(self):
        return self.pembersih._return_df()

    def get_dataframe_course_titles_column(self):
        return self.get_dataframe_after_cleaned_no_duplicate()['course_title']

    def get_dataframe_alamat_column(self):
        return self.get_dataframe_after_cleaned_no_duplicate()['alamat']

    def get_dataframe_nama_column(self):
        nama_alamat = self.get_dataframe_after_cleaned_no_duplicate()['course_title'].str.lower().str.split("#", n=1,expand=True)
        df = pd.DataFrame()
        df['nama'] = nama_alamat[0]
        return df['nama']

    def get_dataframe_subject_column(self):
        return self.get_dataframe_after_cleaned_no_duplicate()['subject']
