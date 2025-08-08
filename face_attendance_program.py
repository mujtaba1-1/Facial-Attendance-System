import cv2
import tkinter as tk
import numpy as np
import PIL.Image, PIL.ImageTk
import time as tm
import ctypes
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import load_model
from tkinter import messagebox
from datetime import datetime, date

# Importing custom modules
from login_page import LoginPage
from face_attendance_gui import FaceAttendanceGUI

class DatabaseFunctions:
    def __init__(self, connection, cursor):
        # Initialise DatabaseFunctions with connection and cursor
        self.connection = connection
        self.cursor = cursor
        self.class_dict = {} # Creates and empty dictionary
        self.graph_open = False # Initialises the graph open as false

        # Sets the time threshold for marking as late
        self.late_time = "14:40:00"
        self.late_time = datetime.strptime(self.late_time, "%H:%M:%S")

        # Calls the method initialise_absence
        self.initialise_absence()

    def populate_class_labels(self):
        # Fetch student IDs and first names from the database
        self.cursor.execute("SELECT student_id, first_name FROM students")
        results = self.cursor.fetchall()

        values = [(result[0], result[1]) for result in results]

        # Stores student IDs as the keys and the first names as values in the dictionary
        for value in values:
            self.class_dict[value[0] -1] = value[1]      
        print(self.class_dict)
        return self.class_dict

    def populate_marked_faces(self):
        marked_face = [] # Creates an empty array 

        # Gets the current day and formats it to use '_'
        today = date.today()
        today = today.strftime("%Y_%m_%d")

        # Fetch the values in the columns: 'today' and student_id
        query = "SELECT {}, {} FROM attendance".format(today, "student_id")
        self.cursor.execute(query)
        results = self.cursor.fetchall()

        values = [(result[0], result[1]) for result in results]

        # Iterates through each of the values
        for value in values:
            # If the value is not equal to A then add teh student id to the array
            if value[0] != 'A':
                marked_face.append(value[1] - 1)

        return marked_face
    
    def initialise_absence(self):
        # Gets the current day and formats it to use '_'
        today = date.today()
        today = today.strftime("%Y_%m_%d")

        # Initialises all the student's attendance as absent for the day
        query = """UPDATE attendance
                   SET {} = '{}'
                """.format(today, 'A')
        self.cursor.execute(query)
        self.connection.commit()

    def calculate_percentage_attendance(self, df):
        # Calculate the total days the student has currently attended
        total_days = (df[df.columns[1:]] != '-').sum(axis=1)
    
        # Count total present and late days, excluding '-'
        df['Total Present'] = df[df != '-'].eq('P').sum(axis=1) + df[df != '-'].eq('L').sum(axis=1)
    
        # Calculate attendance percentage, excluding '-'
        df['Attendance Percentage'] = (df['Total Present'] / total_days) * 100

        return df
    
    def draw_bar_chart(self, df):
        # Checks whether there is already a graph open
        if not self.graph_open:
            self.graph_open = True
            # Plotting
            ax = df.plot(kind='bar', x='full_name', y='Attendance Percentage', color='blue')
            plt.title('Percentage Attendance for Each Student')
            plt.xlabel('Student Name')
            plt.ylabel('Percentage Attendance')
            ax.set_ylim([0, 100])
            ax.set_yticks(range(0, 101, 5))
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
            self.graph_open = False


    def get_student_id(self, first_name, last_name):
        # Seach for the student_id based on the first and last name inputted
        query = "SELECT student_id FROM students WHERE first_name = %s AND last_name = %s"
        self.cursor.execute(query, (first_name, last_name))
        result = self.cursor.fetchone()
        # If a student with the name exists, return their ID, else return None
        if result:
            return result[0]
        else:
            return None
        
    def search_student(self, first_name, last_name):    
        # Calls the method get_student_id and takes two parameters
        student_id = self.get_student_id(first_name, last_name)

        # If a student ID has been found execute the following code
        if student_id:
            self.cursor.execute("DESCRIBE {}".format("attendance")) # Fetches the column names in the database
            columns = [row[0] for row in self.cursor.fetchall()] # Store all of them in an array
    
            date_columns = ', '.join(columns[2:]) # Join the column ,starting from the index 2, names with a ',' to form a string
            
            # Fetch the attendance record for the specific student
            query = """
                    SELECT {} FROM attendance WHERE student_id = {}
                    """.format(date_columns, student_id)
            self.cursor.execute(query)
            results = self.cursor.fetchall()
    
            # Create a dataframe from the results with the date columns from the attendance table
            attendance_df = pd.DataFrame(results, columns=columns[2:])
            attendance_df = attendance_df.transpose()
        
            # Count occurrences of each attendance status
            present_count = attendance_df.eq('P').sum().sum() + attendance_df.eq('L').sum().sum()
            absent_count = attendance_df.eq('A').sum().sum()
            late_count = attendance_df.eq('L').sum().sum()
        
            # Calculate the total number of days
            total_days = attendance_df.shape[0]

            # Calculate the percentage of absent days
            absent_percentage = (absent_count / total_days) * 100

            # Display attendance results in a message box
            messagebox.showinfo("Attendance Results", 
                                f"Total number of days present: {present_count}\n"
                                f"Total number of days late: {late_count}\n"
                                f"Total number of days absent: {absent_count}\n"
                                f"Percentage of absent days for the whole {total_days} days: {absent_percentage:.2f}%")
        
        else:
            # Display an error messagebox if the student is not found
            messagebox.showerror("Error", "Student not found")

    def generate_bar_graph(self):
        # Generate and display a bar graph showing the percentage attendance for each student

        # Fetch the column names from the attendance table
        self.cursor.execute("DESCRIBE {}".format("attendance"))
        columns = [row[0] for row in self.cursor.fetchall()]

        # Join the column names to form a string
        date_columns = ', '.join(columns[2:])

        # Select attendance record for all students
        #   line 1: concatenates the first and last name from the student table as 'full_name'
        #   line 2: specifies the table attendance and uses the alias 'a'
        #   line 3: joins the student and attendance table based on the student_id, linking students to their attendance records
        query = """
                SELECT CONCAT(s.first_name, ' ', s.last_name) AS full_name, {}
                FROM attendance a
                INNER JOIN students s ON a.student_id = s.student_id
                """.format(date_columns)
    
        self.cursor.execute(query)
        results = self.cursor.fetchall()
        
        # Create a dataframe from the results with column names 'full_name' and attendance dates
        attendance_df = pd.DataFrame(results, columns=['full_name'] + columns[2:])
    
        # Calls the method to caluclate the percentage attendance for each student
        attendance_df = self.calculate_percentage_attendance(attendance_df)

        # Calls the method to draw and display the bar chart
        self.draw_bar_chart(attendance_df)

    def markFace(self, ID):
        # Mark the attendance of a student with the given ID

        # Get the current time
        time = datetime.now().time()

        # Get the current date
        today = date.today()
        today = today.strftime("%Y_%m_%d")

        # Fetch the attendance status of the student for the current date
        query = "SELECT {} FROM attendance WHERE student_id = {}".format(today, ID)
        self.cursor.execute(query)
        currentStatus = self.cursor.fetchone()

        # Check if the student's attendance is not marked yet for today
        if currentStatus[0] == 'A':
            # Check if the current time is before the late  time
            if time < self.late_time.time():
                # Update the attendance status to 'P' (Present) if the student is on time
                query = """UPDATE attendance
                           SET {} = '{}'
                           WHERE student_id = {}
                        """.format(today, 'P', ID)
                self.cursor.execute(query)
                self.connection.commit()

            else:
                # Update the attendance status to 'L' (Late) if the student is late
                query = """UPDATE attendance
                           SET {} = '{}'
                           WHERE student_id = {}
                        """.format(today, 'L', ID)
                self.cursor.execute(query)
                self.connection.commit()
 

class FaceAttendance(FaceAttendanceGUI):

    def __init__(self, window, connection, db_functions):
        
        # Set the title and geometry of the tkinter window
        self.window = window
        self.window.title("Facial Attendance System")
        self.window.geometry("1680x1050")
        self.window.resizable(False, False)

        # Set the protocol to handle window closing event
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Initialise the user interface
        self.app_ui()

        # Initialise the database connection and functions
        self.connection = connection
        self.cursor = self.connection.cursor()
        self.db_functions = db_functions

        # Popluate the class labels dictionary and the face_marked array from the database
        self.class_labels = db_functions.populate_class_labels()
        self.faces_marked = db_functions.populate_marked_faces()

        # Calls the method to setup the face detection and recognition models
        self.setup_models()

        self.is_running = True  # Camera is always on
        self.face_detection = False  # Face Detection is off initially
        self.face_recognition = False  # Face Recogntion is off initially
       
        # Set validation parameters
        self.validation_duration = 10
        self.validation_min_count = 20
        self.validation_threshold = 0.8

        # Initialise temporary arrays for face validation
        self.temp_validation_face_1 = []
        self.temp_validation_face_2 = []
        self.temp_validation_face_3 = []
        self.temp_validation_face_4 = []

        # Call the method 'update' 
        self.update()

        # Start the main loop
        self.window.mainloop()
    
    def setup_models(self):
        # Load Face Detection Model
        self.face_detection_model = cv2.dnn.readNetFromCaffe('models/deploy.prototxt', 'models/res10_300x300_ssd_iter_140000.caffemodel')
        # Load Face Recognition Model
        self.face_recognition_model = load_model('models/model.h5')

    def on_closing(self):
        # Handle window closing event

        # Display confirmation dialog and close window if confirmed
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.cursor.close()
            self.connection.close()
            self.window.destroy()
            
    
    def start_face_recognition(self):
        # Sets face detection and recognition flags to true
        self.face_detection = True
        self.face_recognition = True
    
    def stop(self):
        # Sets face detection and recognition flags to false
        self.face_detection = False
        self.face_recognition = False
    
    def update_output(self, output_text):
        # Update the output text area with new text

        self.output_text.config(state=tk.NORMAL) # Enable text area for editing
        self.output_text.insert(tk.END, output_text + "\n\n") # Insert new text at the end
        self.output_text.see(tk.END) # Ensure that the scrollbar is at the bottom
        self.output_text.config(state=tk.DISABLED) # Disable text area for editing

    def clear_output(self):
        # Clears output text area

        self.output_text.config(state=tk.NORMAL) # Enable text area for editing
        self.output_text.delete('1.0', tk.END) # Delete all text
        self.output_text.config(state=tk.DISABLED) # Disable text area for editing

    def detect_faces(self, frame):
        # Detect faces in the frame
        
        # Detect faces using the SSD model
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104, 117, 123))
        self.face_detection_model.setInput(blob)
        detections = self.face_detection_model.forward()
        
        # Iterate over the detected faces and draw rectangles
        for i in range(0, detections.shape[2]):
            if i < 4:
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:  # Filter out weak detections

                    # Get bouding box coordinates
                    box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                    (x, y, x1, y1) = box.astype("int")

                    # call the recognise_faces method if face recognition is enabled
                    if self.face_recognition:
                        self.recognise_faces(frame, x, y, x1, y1, i)
                    else:
                        cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)
        
        return frame
    
    def recognise_faces(self, frame, x, y, x1, y1, face_index):
        # Recognise faces in the frame

        # Extract the detected face
        face = frame[y:y1, x:x1]

        # Preprocess the face for the face recognition model
        face = cv2.resize(face, (224, 224))
        face = np.array(face) / 255.0
        input_batch = np.expand_dims(face, axis=0)

        # Make predictions using the model
        predictions = self.face_recognition_model.predict(input_batch)
        class_indices = np.argmax(predictions, axis=1)
        predicted_class_label = self.class_labels[class_indices[0]]

        if max(predictions[0]) < 0.70:
            # Draw a rectangle and label the face as 'uknown'
            cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)
            cv2.putText(frame, "Uknown Person", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            output_text = (
            f"Class: {predicted_class_label}\n"
            f"Probability: {max(predictions[0]):.2f}\n"
            f"Location: x={x}, y={y}, width={x1-x}, height={y1-y}\n"
            "Detection: Face Detected"
            )

        else:
            # Draw rectangle around the face and label it with the predicted class label
            cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)
            cv2.putText(frame, predicted_class_label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            output_text = (
            f"Class: {predicted_class_label}\n"
            f"{class_indices}\n"
            f"Probability: {max(predictions[0]):.2f}\n"
            f"Location: x={x}, y={y}, width={x1-x}, height={y1-y}\n"
            "Detection: Face Detected"
            )

            #self.update_output(f"face index: {face_index}")

            # If the individual recognised has not been marked call the validation method to mark attendance
            if (class_indices[0] in self.faces_marked) == False:

                if face_index == 0:
                    self.temp_validation_face_1.append((class_indices[0], tm.time()))
                    self.validation(self.temp_validation_face_1, face_index)

                elif face_index == 1:
                    self.temp_validation_face_2.append((class_indices[0], tm.time()))
                    self.validation(self.temp_validation_face_2, face_index)

                elif face_index == 2:
                    self.temp_validation_face_3.append((class_indices[0], tm.time()))
                    self.validation(self.temp_validation_face_3, face_index)

                elif face_index == 3:
                    self.temp_validation_face_4.append((class_indices[0], tm.time()))
                    self.validation(self.temp_validation_face_4, face_index)

        # Update the output text area with detection information
        self.update_output(output_text)


        return frame

    def validation(self, temp_validation_faces, face_index):
        # Validate face for marking attendance

        # Get current time
        current_time = tm.time()
        validation_results = []

        # Iterate over temporary face validation array
        for face in temp_validation_faces:
            # Check if face was detected within validation duration
            if current_time - face[1] <= self.validation_duration:
                validation_results.append(face[0])

        #self.update_output(f"number of faces in array: {len(validation_results)}")

        # Check if minimum count of validated faces is met
        if len(validation_results) >= self.validation_min_count:
            # Count occurrences of each face ID
            values, counts = np.unique(np.array(validation_results), return_counts=True)
            index = np.argmax(counts)
            most_common_face = values[index]

            #self.update_output(f"{validation_results.count(most_common_face) / len(validation_results)}")

            # Check if the most common face is validated with high enough probability
            if validation_results.count(most_common_face) / len(validation_results) >= self.validation_threshold:

                #self.update_output(f"{validation_results}, {most_common_face}")

                # Clear temporary validation array thats being used
                if face_index == 0:
                    self.temp_validation_face_1 = []
                elif face_index == 1:
                    self.temp_validation_face_2 = []
                elif face_index == 2:
                    self.temp_validation_face_3 = []
                elif face_index == 3:
                    self.temp_validation_face_4 = []

                # Append the most common face to the list of marked faces
                self.faces_marked.append(most_common_face)

                # Mark the attendace of the student in the database
                self.db_functions.markFace(int(most_common_face + 1))

                #self.update_output(f"List of marked faces: {self.faces_marked}")


            else:
                # Clear temporary validation array
                if face_index == 0:
                    self.temp_validation_face_1 = []
                elif face_index == 1:
                    self.temp_validation_face_2 = []
                elif face_index == 2:
                    self.temp_validation_face_3 = []
                elif face_index == 3:
                    self.temp_validation_face_4 = []


    def update(self):
        # Method continuoulsy updates and processes video frames

        # Read a frame from the video capture device
        ret, frame = self.vid.read()

        frame = cv2.flip(frame, 1) # Flip the frame horizontally

        # Check if the frame is successfully read and the system is running
        if ret and self.is_running:

            # If face detection enabled, detect faces in the frame
            if self.face_detection: 
                frame = self.detect_faces(frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Resize the frame to fit the canvas
            photo = cv2.resize(frame, (int(self.vid.get(3)), int(self.vid.get(4))))

            # Convert the frame to a format compatible with tkinter
            photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(photo))

            # Display the frame on the canvas
            self.canvas.create_image(29, 29, image=photo, anchor=tk.NW)
            self.canvas.photo = photo

        # Schedule the next frame update after 10 milliseconds
        if self.is_running:
            self.window.after(10, self.update)
    
    def __del__(self):
        # Destructor method to release video capture device when the object is destroyed

        # Release the video caputre device if it is open
        if self.vid.isOpened():
            self.vid.release()


# MAIN CODE
ctypes.windll.shcore.SetProcessDpiAwareness(2)

# Create a Tkinter window for login
login_root = tk.Tk()

# Initialise the login page
login = LoginPage(login_root, DatabaseFunctions, FaceAttendance)
login_root.mainloop()

