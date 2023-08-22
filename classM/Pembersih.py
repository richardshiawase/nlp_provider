class Pembersih:
    def __init__(self,df):
        self.df1 = df
        self._rubah_dataframe_astype_str()
        self._hilangkan_tanda_baca()
        self._kecilkan_tulisan()
    def _kecilkan_tulisan(self):
        self.df4 = self.df3.applymap(str.lower)
    def _hilangkan_tanda_baca(self):
        self.df3 = self.df2.replace(to_replace=['\.','\&',","],value='',inplace=False,regex=True)

    def _rubah_dataframe_astype_str(self):
        self.df2 = self.df1.astype(str)

    def _return_astype_str(self):
        return self.df

    def _return_df(self):
        # strip column header
        self.df5 = self.df4.rename(columns=lambda x: x.strip())
        return self.df5

    def _return_df_master(self):
        return self.df2


