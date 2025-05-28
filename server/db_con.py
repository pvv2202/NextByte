import mysql.connector

host = 'nextbyte-db.ctuise48qz1d.us-east-2.rds.amazonaws.com'
user = 'admin'
pswd = 'Fp$SJE2021g'
database = 'nextbytedb'

# Initialize a variable to hold the database connection
conn = None

try:
    # Attempt to establish a connection to the MySQL database
    conn = mysql.connector.connect(host=host, 
                                   port=3306,
                                   database=database,
                                   user=user,
                                   password=pswd)
    
    # Check if the connection is successfully established
    if conn.is_connected():
        print('Connected to MySQL database')
        
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM Users")
        
        rows = cursor.fetchall()
        for row in rows:
            print(row[1])

except mysql.connector.Error as e:
    # Print an error message if a connection error occurs
    print(e)

finally:
    # Close the database connection in the 'finally' block to ensure it happens
    if conn is not None and conn.is_connected():
        conn.close()
