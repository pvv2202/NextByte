import sqlite3
from dbutils.pooled_db import PooledDB

class SqliteNextByteDB:
    def __init__(self):
        print('setting up pool')
        self.pool = PooledDB(sqlite3, maxconnections=10, mincached=2, maxcached=5, 
                             blocking=True, database='nb.db', check_same_thread=False)
       

    def get_connection(self):
        return self.pool.connection()

    def execute_query(self, query, params=None):
        conn = self.get_connection()
        cursor = conn.cursor()
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        return result
    
    def insert(self, query, params=None):
        conn = self.get_connection()
        cursor = conn.cursor()
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        
        conn.commit()
        cursor.close()
        conn.close()
       
        
        
    





    
  
    
    
