import mysql.connector
from mysql.connector import Error
from datetime import datetime, timedelta

# Function to connect to the MySQL database
def connect_to_database():
    try:
        connection = mysql.connector.connect(
            host="127.0.0.1",
            port=3306,
            user="root",
            password="REPLACE WITH YOUR OWN PASSWORD",
            database="facialrecognition"
        )
        if connection.is_connected():
            print("Connected to MySQL database")
            return connection
    except Error as e:
        print("Error while connecting to database", e)
        return None

# Function to create database tables if they don't exist
def create_tables(connection, start_date, num_days):
    try:
        cursor = connection.cursor()
        # Create students table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS students (
                student_id INT AUTO_INCREMENT PRIMARY KEY,
                first_name VARCHAR(255) NOT NULL,
                last_name VARCHAR(255) NOT NULL,
                date_of_birth DATE NOT NULL
            )
        """)
        connection.commit()
        print("Students table created successfully")

        # Create attendance table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS attendance (
                attendance_id INT AUTO_INCREMENT PRIMARY KEY,
                student_id INT,
                FOREIGN KEY (student_id) REFERENCES students(student_id)
            )
        """)
        connection.commit()
        print("Attendance table created successfully")

        # Calls the function to add the date columns
        insert_attendance_columns(connection, start_date, num_days)
        
    except Error as e:
        print("Error creating tables:", e)

# Function to insert a new student into the database
def insert_student(connection, first_name, last_name, date_of_birth):
    try:
        cursor = connection.cursor()
        # Insert student into the students table
        sql_command = "INSERT INTO students (first_name, last_name, date_of_birth) VALUES (%s, %s, %s)"
        cursor.execute(sql_command, (first_name, last_name, date_of_birth))
        connection.commit()
    except Error as e:
        print("Error inserting student:", e)

# Function to insert date columns into attendance table
def insert_attendance_columns(connection, start_date, num_days):
    try:
        cursor = connection.cursor()
        for i in range(num_days):
            date = start_date + timedelta(days=i)
            column_name = date.strftime('%Y_%m_%d')
            cursor.execute("ALTER TABLE attendance ADD COLUMN %s CHAR DEFAULT '-'" % column_name)
        connection.commit()
        print("Attendance columns added successfully")
    except Error as e:
        print("Error adding attendance columns:", e)

# Function to insert attendance records for a student
def insert_attendance(connection, student_id, start_date, num_days):
    try:
        cursor = connection.cursor()
        cursor.execute("INSERT INTO attendance (attendance_id, student_id) VALUES (%s, %s)", (student_id, student_id))
        connection.commit()
        for i in range(num_days):
            date = start_date + timedelta(days=i)
            column_name = date.strftime('%Y_%m_%d')
            sql_command = "UPDATE attendance SET {} = '{}' WHERE student_id = {}".format(column_name, '-', student_id)
            cursor.execute(sql_command)
        connection.commit()
        print("Attendance records inserted successfully")
    except Error as e:
        print("Error inserting attendance records:", e)

# Connect to the database
connection = connect_to_database()
if connection:
    
    start_date = datetime.now().date()
    num_days = 30

    # Create tables if they don't exist
    create_tables(connection, start_date, num_days)

    # Insert sample student data
    insert_student(connection, "Cristiano", "Ronaldo", '1991-06-15')
    insert_student(connection, "Lionel", "Messi", '1994-09-25')
    insert_student(connection, "Mujtaba", "Butt", '2006-04-01')        
    insert_student(connection, "Neymar", "Jr", '1999-07-09')

    # Insert attendance records for each student
    insert_attendance(connection, 1, start_date, num_days)
    insert_attendance(connection, 2, start_date, num_days)
    insert_attendance(connection, 3, start_date, num_days)
    insert_attendance(connection, 4, start_date, num_days)

    # Close the database connection
    connection.close()

    
