class ColumnOutput():
    def __init__(self):
        self.header = ['Nama', 'Alamat', 'Prediction', 'Alamat_Prediction', 'Score', 'Compared', 'Clean', 'RI', 'RJ']

        pass

    def count_size_header(self):
        return len(self.header)

    def get_header(self):
        return self.header

    def get_column_to_output(self):
        # set column
        return self.header

    def set_column_output(self,list_column):
        self.header = list_column


