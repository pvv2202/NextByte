import mysql.connector

dbconfig = {
    "host": 'nextbyte-db.ctuise48qz1d.us-east-2.rds.amazonaws.com',
    "user": 'admin',
    "password": 'Fp$SJE2021g',
    "database": 'nextbytedb',
    "port": 3306  
}


# Initialize a variable to hold the database connection
conn = None

try:
    # Attempt to establish a connection to the MySQL database
    
    pool = mysql.connector.connect(
        pool_name='nb-pool',
        pool_size=10,
        **dbconfig
    )
    
    cnx = mysql.connector.connect(pool_name='nb-pool')
    cursor = cnx.cursor()
    cursor.execute("SELECT * FROM Users")
    row = cursor.fetchone()
    print(row)
    
    
    


except mysql.connector.Error as e:
    # Print an error message if a connection error occurs
    print(e)

finally:
    # Close the database connection in the 'finally' block to ensure it happens
    if conn is not None and conn.is_connected():
        conn.close()
