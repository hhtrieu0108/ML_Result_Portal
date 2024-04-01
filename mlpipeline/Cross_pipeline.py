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

#Connect to Oracle and Postgresql database
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
    pred = fts.select_all(table='predict_result')
    trans = oracle.select_all(table='transactions')
    #Cross-sell
    tp = transac_product(trans, pred)
    cross = cross_sell(tp)

# Cross sell --> Product Type (Transactions)
def transac_product(trans,predict_result):
    pro_df = trans[['CIF_ID', 'PRODUCT_TYPE']].drop_duplicates()
    pro_df = pro_df.groupby(['CIF_ID'])['PRODUCT_TYPE'].apply(lambda x: x.tolist()).reset_index()
    pro_df['PRODUCT_TYPE'] = [[item.strip() for item in sublst] for sublst in pro_df['PRODUCT_TYPE']]
    pro_df = pd.merge(pro_df, predict_result[['CIF_ID', 'Cluster']], how = 'left', on = 'CIF_ID')
    return pro_df

def cross_sell(pro_df):
    for cluster_id in pro_df['Cluster'].unique():
        # Lọc dữ liệu cho cluster hiện tại
        cluster_data = pro_df[pro_df['Cluster'] == cluster_id]
        # Thêm sort value theo PRODUCT_TYPE:
        cluster_data = cluster_data.sort_values(by = 'CIF_ID', ascending = True)
        cluster_data = cluster_data.sort_values(by = 'PRODUCT_TYPE', ascending = True)
        rules_data = cluster_data['PRODUCT_TYPE']
        te = TransactionEncoder()
        te_ary = te.fit(rules_data).transform(rules_data)
        fpgrowth_df = pd.DataFrame(te_ary, columns=te.columns_)
        frequent_patterns = fpgrowth(fpgrowth_df, min_support=0.0001, use_colnames=True)
        df_rules = association_rules(frequent_patterns, metric="confidence", min_threshold=0.5)
        df_rules['antecedents'] = df_rules['antecedents'].apply(list)
        df_rules['consequents'] = df_rules['consequents'].apply(list)
        df_rules = df_rules[df_rules['confidence'] >= 0.6]
        df_rules = df_rules[df_rules['lift'] > 1]
        df_rules = df_rules.sort_values(by = ['confidence', 'lift'], ascending = False)
        list_product= list(df_rules['antecedents'])
        list_recommend= list(df_rules['consequents'])
        def create_recommendation(row):
            PRODUCT_TYPE = row['PRODUCT_TYPE']
            recommendations = []
            for i, value in enumerate(list_product):
                if value == PRODUCT_TYPE:
                    recommendations.append(list_recommend[i])
            if recommendations:
                return recommendations
            else:
                return ['No Recommend']
        pro_df.loc[pro_df['Cluster'] == cluster_id, 'Recommend'] = cluster_data.apply(create_recommendation, axis=1)
    fts.export_table(data=pro_df,table_name='cross_sell')
    return pro_df

main()
