import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules
import warnings
warnings.filterwarnings("ignore")

km = joblib.load('kmeans_segment.joblib')
rf_model = joblib.load('randomforest.joblib')
up_deb_gold = joblib.load('Upsell_DebitGold_model.joblib')
up_deb_plat = joblib.load('Upsell_DebitPlatinum_model.joblib')
up_cred_gold = joblib.load('Upsell_CreditGold_model.joblib')
up_cred_plat = joblib.load('Upsell_CreditPlatinum_model.joblib')

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
    deb = oracle.select_all(table='debit')
    cred = oracle.select_all(table='credit')
    pred = fts.select_all(table='predict_result')
    deb_std, deb_gld = prepare_debit(deb, pred)
    cred_std, cred_gld = prepare_credit(cred, pred)
    #Up-sell-Debit
    deb_gold_pred = upsell_deb_gold(deb_std)
    deb_plat_pred = upsell_deb_plat(deb_gld)
    #Up-sell-Credit
    cred_gold_pred = upsell_cred_gold(cred_std)
    cred_plat_pred = upsell_cred_plat(cred_gld)

def prepare_debit(deb,seg_result):
    deb = deb[['CIF_ID','CARD_CLASS']].drop_duplicates()
    card_cus = deb.groupby(['CIF_ID'])['CARD_CLASS'].apply(lambda x: x.tolist()).reset_index()
    card_cus['CARD_CLASS'] = [[item.strip() for item in sublst] for sublst in card_cus['CARD_CLASS']]
    card_cus = pd.merge(card_cus, seg_result[['CIF_ID', 'Cluster','Recency','Frequency','Monetary']], how = 'left', on = 'CIF_ID')
    def encode_card_type(row):
        card_type = row['CARD_CLASS']
        encoded =  {
            'Debit_Standard' : 0,
            'Debit_Gold': 0,
            'Debit_Platinum': 0
        }
        for card in card_type:
            if 'Standard' in card:
                encoded['Debit_Standard'] = 1
            elif 'Gold' in card:
                encoded['Debit_Gold'] = 1
            elif 'Platinum' in card:
                encoded['Debit_Platinum'] = 1
        return pd.Series(encoded)
    encode_card_types = card_cus.apply(encode_card_type, axis=1)
    card_cus = pd.concat([card_cus, encode_card_types],axis=1)
    debit_standard = card_cus.loc[(card_cus['Debit_Standard'] == 1) & (card_cus['Debit_Gold'] == 0) & (card_cus['Debit_Platinum'] == 0)]
    debit_standard = debit_standard.drop(columns=['Debit_Gold','Debit_Platinum'])
    debit_gold = card_cus.loc[(card_cus['Debit_Gold'] == 1) & (card_cus['Debit_Platinum'] == 0)]
    debit_gold = debit_gold.drop(columns=['Debit_Standard','Debit_Platinum'])
    # fts.export_table(data=debit_standard,table_name='prepare_debit_standard')
    # fts.export_table(data=debit_gold,table_name='prepare_debit_gold')
    return debit_standard, debit_gold

def upsell_deb_gold(debit_standard):
    deb_pred = debit_standard.drop(columns=['CIF_ID', 'CARD_CLASS'])
    deb_pred_scaled = scaler.fit_transform(deb_pred)
    y_pred_proba = up_deb_gold.predict_proba(deb_pred_scaled)
    y_pred_class = up_deb_gold.predict(deb_pred_scaled)
    up_deb_pred = pd.DataFrame({
        'Upsell_to_Gold': y_pred_class,
        'Probability_of_Yes': y_pred_proba[:, 1],
        'Probability_of_No': y_pred_proba[:, 0]
    })
    up_deb_gld_res = pd.concat([debit_standard.reset_index(drop=True), up_deb_pred], axis=1)
    up_deb_gld_res.loc[up_deb_gld_res['Upsell_to_Gold'] == 1, 'Upsell_to_Gold'] = 'Yes'
    up_deb_gld_res.loc[up_deb_gld_res['Upsell_to_Gold'] == 0, 'Upsell_to_Gold'] = 'No'
    up_deb_gld_res.drop(columns='Debit_Standard', inplace=True)
    fts.export_table(data=up_deb_gld_res,table_name='upsell_debit_gold')
    return up_deb_gld_res

def upsell_deb_plat(debit_gold):
    deb_pred = debit_gold.drop(columns=['CIF_ID','CARD_CLASS'])
    deb_pred_scaled = scaler.fit_transform(deb_pred)
    y_pred_proba = up_deb_plat.predict_proba(deb_pred_scaled)
    y_pred_class = up_deb_plat.predict(deb_pred_scaled)
    up_deb_pred = pd.DataFrame({
        'Upsell_to_Platinum': y_pred_class,
        'Probability_of_Yes': y_pred_proba[:, 1],
        'Probability_of_No': y_pred_proba[:,0]
    })
    up_deb_plt_res = pd.concat([debit_gold.reset_index(drop=True), up_deb_pred], axis=1)
    up_deb_plt_res.loc[up_deb_plt_res['Upsell_to_Platinum'] == 1, 'Upsell_to_Platinum'] = 'Yes'
    up_deb_plt_res.loc[up_deb_plt_res['Upsell_to_Platinum'] == 0, 'Upsell_to_Platinum'] = 'No'
    up_deb_plt_res.drop(columns='Debit_Gold', inplace=True)
    fts.export_table(data=up_deb_plt_res,table_name='upsell_debit_plat')
    return up_deb_plt_res

# Up sell --> Credit
def prepare_credit(cred, seg_result):
    cred = cred[['CIF_ID','CARD_CLASS']].drop_duplicates()
    card_cus = cred.groupby(['CIF_ID'])['CARD_CLASS'].apply(lambda x: x.tolist()).reset_index()
    card_cus['CARD_CLASS'] = [[item.strip() for item in sublst] for sublst in card_cus['CARD_CLASS']]
    card_cus = pd.merge(card_cus, seg_result[['CIF_ID', 'Cluster','Recency','Frequency','Monetary']], how = 'left', on = 'CIF_ID')
    def encode_card_type(row):
        card_type = row['CARD_CLASS']
        encoded = {
            'Credit_Standard': 0,
            'Credit_Gold': 0,
            'Credit_Platinum': 0
        }
        for card in card_type:
            if 'Standard' in card:
                encoded['Credit_Standard'] = 1
            elif 'Gold' in card:
                encoded['Credit_Gold'] = 1
            elif 'Platinum' in card:
                encoded['Credit_Platinum'] = 1
        return pd.Series(encoded)
    encode_card_types = card_cus.apply(encode_card_type, axis=1)
    card_cus = pd.concat([card_cus, encode_card_types],axis=1)
    credit_standard = card_cus.loc[(card_cus['Credit_Standard'] == 1) & (card_cus['Credit_Gold'] == 0) & (card_cus['Credit_Platinum'] == 0)]
    credit_standard = credit_standard.drop(columns=['Credit_Gold','Credit_Platinum'])
    credit_gold = card_cus.loc[(card_cus['Credit_Gold'] == 1) & (card_cus['Credit_Platinum'] == 0)]
    credit_gold = credit_gold.drop(columns=['Credit_Standard','Credit_Platinum'])
    # fts.export_table(data=credit_standard,table_name='prepare_credit_standard')
    # fts.export_table(data=credit_gold,table_name='prepare_credit_gold')
    return credit_standard, credit_gold

def upsell_cred_gold(credit_standard):
    cred_pred = credit_standard.drop(columns=['CIF_ID','CARD_CLASS'])
    cred_pred_scaled = scaler.fit_transform(cred_pred)
    y_pred_proba = up_cred_gold.predict_proba(cred_pred_scaled)
    y_pred_class = up_cred_gold.predict(cred_pred_scaled)
    up_cred_pred = pd.DataFrame({
        'Upsell_to_Gold': y_pred_class,
        'Probability_of_Yes': y_pred_proba[:, 1],
        'Probability_of_No': y_pred_proba[:, 0]
    })
    up_cred_gld_res = pd.concat([credit_standard.reset_index(drop=True), up_cred_pred], axis=1)
    up_cred_gld_res['Upsell_to_Gold'] = up_cred_gld_res['Upsell_to_Gold'].map({1: 'Yes', 0: 'No'})
    up_cred_gld_res.drop(columns='Credit_Standard', inplace=True)
    fts.export_table(data=up_cred_gld_res,table_name='upsell_cred_gold')
    return up_cred_gld_res

def upsell_cred_plat(credit_gold):
    cred_pred = credit_gold.drop(columns=['CIF_ID','CARD_CLASS'])
    cred_pred_scaled = scaler.fit_transform(cred_pred)
    y_pred_proba = up_cred_plat.predict_proba(cred_pred_scaled)
    y_pred_class = up_cred_plat.predict(cred_pred_scaled)
    up_cred_pred = pd.DataFrame({
        'Upsell_to_Platinum': y_pred_class,
        'Probability_of_Yes': y_pred_proba[:, 1],
        'Probability_of_No': y_pred_proba[:, 0]
    })
    up_cred_plt_res = pd.concat([credit_gold.reset_index(drop=True), up_cred_pred], axis=1)
    up_cred_plt_res['Upsell_to_Platinum'] = up_cred_plt_res['Upsell_to_Platinum'].map({1: 'Yes', 0: 'No'})
    up_cred_plt_res.drop(columns='Credit_Gold', inplace=True)
    fts.export_table(data=up_cred_plt_res,table_name='upsell_cred_plat')
    return up_cred_plt_res

main()