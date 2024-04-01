import os
from flask_cors import CORS
from flask import Flask, request, render_template, make_response, jsonify, send_from_directory, send_file, Response, redirect, url_for

HOST = "0.0.0.0"
PORT_NUMBER = int(os.getenv('PORT_NUMBER', 7000))
APP_ROOT_1 = os.getenv('APP_ROOT', '/infer1')

app = Flask(__name__)
CORS(app)

#Connect to Postgresql database
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

psg = Postgre_Import(host='180.93.172.220',port='5432',uid='data_team',pwd='4Eglqghe8TMzxCMy5G23T',database='Output')
psg.connect_postgre_sql()

@app.route('/')
def home():
    return render_template('home.html')

# Show ML Pipeline
@app.route('/get_ml_pipeline', methods=['GET'])
def get_ml_pipeline():
    try:
        pipeline = send_from_directory('static', 'pipeline.png', mimetype= 'image/png')
        return pipeline
    except Exception as e:
        print(f'An error occurred: {str(e)}')
        return make_response(jsonify({'error': 'Internal Server Error'}),500)


# Show Segment
@app.route('/segment_page')
def segment_page():
    try:
        seg_pd = psg.select_all(table='segment_result')
        seg_pd['Cluster'] = seg_pd['Cluster'].astype('int')
        pred = seg_pd.head(100).to_dict(orient='records')
        return render_template('segment_page.html', data=pred)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return make_response(jsonify({'error': 'Internal Server Error'}), 500)

@app.route('/download_segment_data')
def download_segment_data():
    seg_pd = psg.select_all(table='segment_result')
    seg_pd['Cluster'] = seg_pd['Cluster'].astype('int')
    resp = make_response(seg_pd.to_csv(index=False))
    resp.headers["Content-Disposition"] = "attachment; filename=segment_data.csv"
    resp.headers["Content-Type"] = "text/csv"
    return resp


# Show Cross_sell
@app.route('/cross_sell_page', methods=['POST', 'GET'])  # Updated route name for clarity
def cross_page():
    try:
        cross = psg.select_all(table='cross_sell')
        cross_pd = cross.sort_values(by='Recommend', ascending=False).head(100)  # Sort and limit rows
        cross_pd_dict = cross_pd.to_dict(orient='records')
        return render_template('cross_page.html', data=cross_pd_dict)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return make_response(jsonify({'error': 'Internal Server Error'}), 500)

@app.route('/download_cross_sell_data')
def download_cross_sell_data():
    cross = psg.select_all(table='cross_sell')
    cross_pd = cross.sort_values(by='Recommend', ascending=False)
    cross_pd = cross_pd.to_csv(index=False)
    return Response(
        cross_pd,
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=cross_sell_data.csv"}
    )


@app.route('/upsell_page')
def upsell_page():
    return render_template('upsell_page.html')


# Show Upsell_Debit
@app.route('/upsell_debit_page', methods=['GET'])
def upsell_debit_page():
    dataset_type = request.args.get('type', 'gold')
    try:
        if dataset_type == 'gold':
            data = psg.select_all(table='upsell_debit_gold')
        elif dataset_type == 'plat':
            data = psg.select_all(table='upsell_debit_plat')
        else:
            return make_response(jsonify({'error': 'Invalid dataset type'}), 400)
        data_pd = data.head(100).to_dict(orient='records')
        return render_template('upsell_debit_page.html', data=data_pd, type=dataset_type)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return make_response(jsonify({'error': 'Internal Server Error'}), 500)


@app.route('/download_upsell_debit')
def download_upsell_debit():
    dataset_type = request.args.get('type', 'gold')
    try:
        if dataset_type == 'gold':
            data = psg.select_all(table='upsell_debit_gold')
        elif dataset_type == 'plat':
            data = psg.select_all(table='upsell_debit_plat')
        else:
            return make_response(jsonify({'error': 'Invalid dataset type'}), 400)
        csv = data.to_csv(index=False)
        return Response(
            csv,
            mimetype="text/csv",
            headers={"Content-disposition": f"attachment; filename=upsell_debit_{dataset_type}.csv"}
        )
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return make_response(jsonify({'error': 'Internal Server Error'}), 500)


# Show Upsell_Credit
@app.route('/upsell_credit_page', methods=['GET'])
def upsell_credit_page():
    dataset_type = request.args.get('type', 'gold')
    try:
        if dataset_type == 'gold':
            data = psg.select_all(table='upsell_cred_gold')
        elif dataset_type == 'plat':
            data = psg.select_all(table='upsell_cred_plat')
        else:
            return make_response(jsonify({'error': 'Invalid dataset type'}), 400)
        data_pd = data.head(100).to_dict(orient='records')
        return render_template('upsell_credit_page.html', data=data_pd, type=dataset_type)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return make_response(jsonify({'error': 'Internal Server Error'}), 500)


@app.route('/download_upsell_credit')
def download_upsell_credit():
    dataset_type = request.args.get('type', 'gold')
    try:
        if dataset_type == 'gold':
            data = psg.select_all(table='upsell_cred_gold')
        elif dataset_type == 'plat':
            data = psg.select_all(table='upsell_cred_plat')
        else:
            return make_response(jsonify({'error': 'Invalid dataset type'}), 400)
        csv = data.to_csv(index=False)
        return Response(
            csv,
            mimetype="text/csv",
            headers={"Content-disposition": f"attachment; filename=upsell_credit_{dataset_type}.csv"}
        )
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return make_response(jsonify({'error': 'Internal Server Error'}), 500)


# Run Portal
if __name__ == '__main__':
    app.run(host=HOST, port=PORT_NUMBER)