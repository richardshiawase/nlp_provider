import pandas as pd


class ExcelBacaTulis:
    def __init__(self):
        pass
    def baca_excel(self,file_excel):
        self.df = pd.read_excel(file_excel)
        return self.df

    def write_to_excel(self,nama,prefix_subject_name,dataframe):
        # # # Declare write
        writer = pd.ExcelWriter('media/' + nama + prefix_subject_name + ".xlsx",
                                engine='xlsxwriter')
        # # # Concat list of dataframe
        full_dfw = pd.concat(list(dataframe), ignore_index=True)

        # # # Convert the dataframe to an XlsxWriter Excel object.
        full_dfw.to_excel(writer, sheet_name='Sheet1', index=False)

        # # # Close the Pandas Excel writer and output the Excel file.
        writer.close()