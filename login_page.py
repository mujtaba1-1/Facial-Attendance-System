import mysql.connector
import tkinter as tk
import os

from tkinter import messagebox

class LoginPage:

    def __init__(self, root, db, faceAttendance):
        # Initialise the login page
        self.root = root
        self.root.title("Database Connection") # Set window title
        self.root.geometry("448x327") # Set window size
        self.root.resizable(False, False) # Disable window resizing

        # Stores database and faceAttendance class references
        self.db = db
        self.faceAttendance = faceAttendance

        # Create the login page UI
        self.login_ui()

    def login_ui(self):
        # Creates canvas
        self.canvas = tk.Canvas(
            self.root,
            bg = "#FFFFFF",
            height=327,
            width=448,
            bd=0,
            highlightthickness=0,
            relief="ridge"
            )
        self.canvas.place(x=0, y=0)

        # Loads the background images
        self.bg_1 = tk.PhotoImage(file=os.path.join('assets', 'login gui', 'image_1.png'))
        self.image_1 = self.canvas.create_image(
            224.0,
            163.0,
            image = self.bg_1
            )

        self.bg_2 = tk.PhotoImage(file=os.path.join('assets', 'login gui', 'image_2.png'))
        self.image_2 = self.canvas.create_image(
            224.0, 
            269.0,
            image = self.bg_2
            )

        # Creates the entry box for the password
        self.entry_image_1 = tk.PhotoImage(file=os.path.join('assets', 'login gui', 'entry_1.png'))
        self.entry_bg_1 = tk.Label(
            self.root,
            borderwidth=0,
            highlightthickness=0,
            image=self.entry_image_1
        )
        self.entry_bg_1.place(x = 113.0, y = 142.0)

        self.entry_1 = tk.Entry(
            self.root,
            bd=0,
            bg="#1B2956",
            fg="white",
            insertbackground="white",
            font=("Bahnschrift SemiBold SemiConden", 10),
            highlightthickness=0
        )
        self.entry_1.place(
            x=116.0,
            y=145.0,
            width=215.0,
            height=28.0
        )

        # Creates the entry box for the username
        self.entry_image_2 = tk.PhotoImage(file=os.path.join('assets', 'login gui', 'entry_2.png'))
        self.entry_bg_2 = tk.Label(
            self.root,
            borderwidth=0,
            highlightthickness=0,
            image=self.entry_image_2
        )
        self.entry_bg_2.place(x = 113.0, y = 78.0)

        self.entry_2 = tk.Entry(
            self.root,
            bd=0,
            bg="#1B2956",
            fg="white",
            insertbackground="white",
            font=("Bahnschrift SemiBold SemiConden", 10),
            highlightthickness=0
        )

        self.entry_2.place(
            x=116.0,
            y=81.0,
            width=215.0,
            height=28.0
        )

        # Creates the button to submit the details entered by the user
        self.button_image_1 = tk.PhotoImage(file=os.path.join('assets', 'login gui', 'button_1.png'))
        self.button_1 = tk.Button(
            image=self.button_image_1,
            borderwidth=0,
            highlightthickness=0,
            command=self.on_submit,  # Calls the on_submit method when clicked
            activebackground='#0B194A',
            activeforeground='#0B194A',
            relief="flat"
        )
        self.button_1.place(
            x=174.0,
            y=199.0,
            width=99.0,
            height=29.0
        )

        # Adds the texts seen on the screen
        self.canvas.create_text(
            114.0,
            127.0,
            anchor="nw",
            text="Password",
            fill="#5672A1",
            font=("Bahnschrift SemiBold SemiConden", 10 * -1)
        )

        self.canvas.create_text(
            114.0,
            61.0,
            anchor="nw",
            text="Username",
            fill="#5672A1",
            font=("Bahnschrift SemiBold SemiConden", 10 * -1)
        )

        self.canvas.create_text(
            175.0,
            18.0,
            anchor="nw",
            text="Login to database",
            fill="#FFFFFF",
            font=("Bahnschrift SemiBold SemiConden", 15 * -1)
        )

    def connect_to_database(self, host, port, user, password, database):
        # Attempt to connect to the MySQL databsae
        try:
            connection = mysql.connector.connect(
                host = host,
                port = port,
                user = user,
                password = password,
                database = database
            )
            messagebox.showinfo("Success", "Connected to the database successfully!") # Display the success message

            return connection, True # Return connection object and True if successfully connected

        # In case any errors occurs whilst connection
        except mysql.connector.Error as err:
            messagebox.showerror("Error", f"Failed to connect to the database: {err}") # Display error message if entered details are incorrect
            return None, False # Return None and False for failed connection

    def on_submit(self): # Function that is called when login button is clicked
        entered_password = self.entry_1.get() # Get entered password
        entered_username = self.entry_2.get() # Get entered username

        # Attempts database connection by calling the connect_to_database method
        connection, connected = self.connect_to_database("127.0.0.1", 3306, entered_username, entered_password, "facialrecognition")
        if connected:
            self.root.destroy() # Close the login window
            db_functions = self.db(connection, connection.cursor()) # Create database function instance
            app_root = tk.Tk() # Createa a Tkinter window for the main program
            app = self.faceAttendance(app_root, connection, db_functions) # Create face attendance app instance
