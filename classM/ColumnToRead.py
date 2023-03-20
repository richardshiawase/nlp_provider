class ColumnToRead():
    def __init__(self):
        self.header = ['Nama', 'Alamat', 'Alamat_Prediction', 'RI', 'RJ']

    def count_size_header(self):
        return len(self.header)

    def get_header(self):
        return self.header


    def get_column_to_read_when_process(self):
        # set column
        return self.header

    def set_column_to_read_when_process(self,list_column):
        self.header = list_column

