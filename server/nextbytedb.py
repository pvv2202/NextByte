import mysql.connector
from mysql.connector import pooling

class MySQLNextByteDB:
    def __init__(self):
        self.config = { 
            "host": 'nextbyte-db.ctuise48qz1d.us-east-2.rds.amazonaws.com',
            "user": 'admin',
            "password": 'Fp$SJE2021g',
            "database": 'nextbytedb',
            "port": 3306
        }
        self.pool_name = 'nb-pool'
        self.pool_size = 10
        self.pool = pooling.MySQLConnectionPool(
            pool_name=self.pool_name,
            pool_size=self.pool_size,
            **self.config
        )

    def get_connection(self):
        return self.pool.get_connection()

    def execute_query(self, query):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(query)
        result = cursor.fetchall()
        cursor.close()
        conn.close()
        return result
        
        
    





    
    # cnx = mysql.connector.connect(pool_name='nb-pool')
    # cursor = cnx.cursor()
    # cursor.execute("SELECT * FROM Users")
    # row = cursor.fetchone()
    # print(row)
    
    
