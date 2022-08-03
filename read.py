import sys
import psycopg2
import pandas as pd
import pickle
from sqlalchemy import create_engine


# Connection parameters
param_dic = {
    "host": "18.206.131.186",
    "database": "blockrecondb",
    "port": 5432,
    "user": "vincilro",
    "password": "vincilro1"
}

def postgresql_to_dataframe(table_name):
    """
    Tranform a SELECT query into a pandas dataframe
    """
    uri = f'postgresql+psycopg2://{param_dic["user"]}:{param_dic["password"]}@{param_dic["host"]}:{param_dic["port"]}/{param_dic["database"]}'
    try:
        engine = create_engine(uri)
        table_df = pd.read_sql_table(table_name, con=engine)
    except (Exception, psycopg2.DatabaseError) as error:
        print("Error: %s" % error)
        return 1
    return table_df
