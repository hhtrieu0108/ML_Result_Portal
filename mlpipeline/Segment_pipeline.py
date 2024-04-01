import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules
import warnings
warnings.filterwarnings("ignore")

km = joblib.load('kmeans_segment.joblib')
rf_model = joblib.load('randomforest.joblib')

scaler = StandardScaler()

class Orcle_Import:
    def __init__(self,server,uid,pwd):
        self.server = server
        self.uid = uid
        self.df = None
        self.pwd = pwd
        self.table = None
        self.conn = None
        self.columns = '*'
    def connect_oracle_sql(self):
        import cx_Oracle
        connection = cx_Oracle.connect(self.uid, self.pwd, self.server)
        self.conn = connection
    def select_all(self,columns="*",table=any):
        import pandas as pd 
        self.columns = columns
        self.table = table
        sql_query = f"select {self.columns} from {self.table}"
        df = pd.read_sql(sql_query, self.conn)
        return df
    def close_connect(self):
        self.conn.close()

class Postgre_Import:
    def __init__(self,database,uid,pwd,host,port):
        self.database = database
        self.uid = uid
        self.pwd = pwd
        self.host = host
        self.port = port
        self.conn = None
        self.columns = None
        self.table = None
    def connect_postgre_sql(self):
        import psycopg2
        conn = psycopg2.connect(dbname=self.database, user=self.uid, password=self.pwd, host=self.host, port=self.port)
        self.conn = conn
    def select_all(self,columns="*",table=any):
        import pandas as pd 
        self.columns = columns
        self.table = table
        sql_query = f'SELECT {self.columns} FROM {self.table}'
        # Read the SQL query result into a DataFrame
        df = pd.read_sql(sql_query, self.conn)
        return df
    def post_pipeline(self):
        self.connect_postgre_sql()
        df = self.select_all(table='segment_result')
        return df
    def export_table(self,data,table_name):
        from sqlalchemy import create_engine
        if self.conn is None:
            self.connect_postgre_sql()
        # Sử dụng pandas để tạo bảng và nhập dữ liệu vào PostgreSQL
        engine = create_engine(f'postgresql+psycopg2://{self.uid}:{self.pwd}@{self.host}:{self.port}/{self.database}')
        data.to_sql(name=table_name, con=engine, index=False, if_exists='replace')
        print(f"Table '{table_name}' created and data imported successfully.")
    def close_connect(self):
        self.conn.close()

oracle = Orcle_Import(uid='ERWIN_TEST',pwd='ERWIN_TEST',server='180.93.172.210:1521/XEPDB1')
oracle.connect_oracle_sql()

fts = Postgre_Import(host='180.93.172.220',port='5432',uid='data_team',pwd='4Eglqghe8TMzxCMy5G23T',database='Predict')
fts.connect_postgre_sql()

def main():
    # cus = oracle.select_all(table='cif')
    trans = oracle.select_all(table='transactions')

    rfm_df  =  rfm_calc(trans)
    # master = merge_rfm_cus(cus,rfm_df)
    pred = predict(rfm_df)
    # pred_save = save_result(master,pred)
    oracle.close_connect()
    fts.close_connect()

def rfm_calc(transaction_data):
    transaction_data['TIMESTAMP'] = pd.to_datetime(transaction_data['TIMESTAMP'])
    # Calculate Recency
    rec = transaction_data.groupby(by='CIF_ID', as_index=False)['TIMESTAMP'].max()
    rec.columns = ['CIF_ID', 'LastPurchaseDate']
    rec['LastPurchaseDate'] = rec['LastPurchaseDate'].dt.date
    recent_date = rec['LastPurchaseDate'].max()
    rec['Recency'] = rec['LastPurchaseDate'].apply(lambda x: (recent_date - x).days)
    # Calculate Frequency
    freq = transaction_data.drop_duplicates().groupby(by=['CIF_ID'], as_index=False)['TIMESTAMP'].count()
    freq.columns = ['CIF_ID', 'Frequency']
    # Calculate Monetary
    mone = transaction_data.groupby(by='CIF_ID', as_index=False)['TRANSACTION_AMOUNT'].sum()
    mone.columns = ['CIF_ID', 'Monetary']
    # Merge all
    rfm_df = rec.merge(freq, on='CIF_ID').merge(mone, on='CIF_ID').drop(columns='LastPurchaseDate')
    return rfm_df

# def merge_rfm_cus(cus,rfm_df):
#     cus['CIF_ID'] = cus['CIF_ID'].astype('str')    
#     master = cus.merge(rfm_df,how='left',on='CIF_ID')
#     master.fillna(0,inplace=True)
#     return master

def predict(rfm_df):
    rf_fts = rfm_df[['Recency','Frequency','Monetary']]
    rf_fts_scale = scaler.fit_transform(rf_fts)
    y_pred = km.fit_predict(rf_fts_scale).tolist()
    segment = pd.DataFrame(y_pred,columns=['Cluster'])
    predict_result = pd.concat([rfm_df,segment],axis=1)
    fts.export_table(data=predict_result,table_name='predict_result')
    return predict_result

# def save_result(master, predict_result):
#     segment_result = pd.merge(master,predict_result[['CIF_ID','Cluster']],how='left',on='CIF_ID')
#     segment_result['Cluster'].fillna('Not done any Transactions yet')
#     segment_result.fillna(0,inplace=True)
#     fts.export_table(data=segment_result,table_name='segment_result')
#     return segment_result

main()