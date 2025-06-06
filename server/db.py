import mysql.connector
from mysql.connector import pooling

class MySQLNextByteDB:
    def __init__(self):
        print('connecting to aws database: nextbytedb')
        self.config = { 
            "host": '127.0.0.1',
            "user": 'admin',
            "password": 'Fp$SJE2021g',
            "database": 'nextbytedb',
            "port": 3306
        }
        self.pool_name = 'nb-pool'
        self.pool_size = 1
        self.pool = pooling.MySQLConnectionPool(
            pool_name=self.pool_name,
            pool_size=self.pool_size,
            **self.config
        )

    def get_connection(self):
        return self.pool.get_connection()

    def execute_query(self, query, params=None):
        conn = self.get_connection()
        cursor = conn.cursor(dictionary=True)
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        return result
        
        
    





    
    # cnx = mysql.connector.connect(pool_name='nb-pool')
    # cursor = cnx.cursor()
    # cursor.execute("SELECT * FROM Users")
    # row = cursor.fetchone()
    # print(row)
    
    
