import pandas as pd

class DataPreprocessing:
    
    def __init__(self, train_data, transaction_data, user_logs_data, members_data) -> None:
        self.train_data = train_data
        self.transaction_data = transaction_data
        self.user_logs_data = user_logs_data
        self.members_data = members_data

    def drop_duplicates(self):
        self.train_data.drop_duplicates(subset=['msno'], inplace=True)
        self.transaction_data.drop_duplicates(subset='msno', inplace=True)
        self.user_log_data.drop_duplicates(subset='msno', inplace=True)
        self.members_data.drop_duplicates(subset='msno', inplace=True)

    def drop_null_values(self):
        self.train_data.dropna(subset=['msno'], inplace=True)
        self.transaction_data.dropna(subset=['msno'], inplace=True)
        self.user_log_data.dropna(subset=['msno'], inplace=True)
        self.members_data.dropna(subset=['msno'], inplace=True)

    def convert_to_data_format(self):
        self.transaction_data.transaction_date = pd.to_datetime(
            self.transaction_data['transaction_date'], format='%Y%m%d')
        self.transaction_data.membership_expire_date = pd.to_datetime(
            self.transaction_data['membership_expire_date'], format='%Y%m%d')
        self.user_logs_data.date = pd.to_datetime(
            self.user_logs_data['date'], format='%Y%m%d')
        self.members_data.expiration_date = pd.to_datetime(
            self.members_data['expiration_date'], format='%Y%m%d')
        self.members_data.registration_init_time = pd.to_datetime(
            self.members_data['registration_init_time'], format='%Y%m%d')

    def data_normalization(self):
        from sklearn.preprocessing import MinMaxScaler
        norm_transaction_data = pd.DataFrame(
            MinMaxScaler().fit_transform(self.transaction_data))
        norm_user_logs_data = pd.DataFrame(
            MinMaxScaler().fit_transform(self.user_logs_data))
        norm_members_data = pd.DataFrame(
            MinMaxScaler().fit_transform(self.members_data))
