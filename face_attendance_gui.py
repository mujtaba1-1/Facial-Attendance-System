import tkinter as tk
import os
import cv2

class FaceAttendanceGUI:       
    def app_ui(self):
        # Create the main window
        self.canvas = tk.Canvas(
            self.window,
            bg='#FFFFFF',
            height=1050,
            width=1680,
            bd=0,
            highlightthickness=0,
            relief='ridge'
            )
        self.canvas.place(x=0, y=0)

        # Load and place background images
        self.bg_1 = tk.PhotoImage(file=os.path.join('assets', 'main gui', 'image_1.png'))
        self.image_1 = self.canvas.create_image(
            840.0,
            525.0,
            image = self.bg_1
            )

        self.bg_2 = tk.PhotoImage(file=os.path.join('assets', 'main gui', 'image_2.png'))
        self.image_2 = self.canvas.create_image(
            840.0,
            945.0,
            image = self.bg_2
            )

        # Load and place reset button
        self.reset_image_1 = tk.PhotoImage(file=os.path.join('assets', 'main gui', 'button_1.png'))
        self.reset_bt = tk.Button(
            image = self.reset_image_1,
            borderwidth=0,
            highlightthickness=0,
            command= self.clear_output, # Calls the clear_output method when clicked
            activebackground='#0B194A',
            activeforeground='#0B194A',
            relief="flat"
            )
        self.reset_bt.place(
            x=1365.0,
            y=774.0,
            width=251.0,
            height=69.0
            )

        # Creates box for the camera to be placed in
        self.canvas.create_rectangle(
            25.0,
            31.0,
            1305.0,
            751.0,
            fill="#1B2956",
            outline="")

        # Creates multiple line UI elements
        self.canvas.create_rectangle(
            1328.94287109375,
            31.00152587890625,
            1652.94287109375,
            32.00152587890625,
            fill="#1B2956",
            outline="")

        self.canvas.create_rectangle(
            1328.0,
            30.0,
            1329.0,
            752.0027465820312,
            fill="#1B2956",
            outline="")

        self.canvas.create_rectangle(
            1328.94287109375,
            751.0015258789062,
            1652.94287109375,
            752.0015258789062,
            fill="#1B2956",
            outline="")

        self.canvas.create_rectangle(
            1651.94287109375,
            30.00152587890625,
            1652.94287109375,
            752.0015258789062,
            fill="#1B2956",
            outline="")

        # Load and place output text area
        self.output_image_1 = tk.PhotoImage(
            file=os.path.join('assets', 'main gui', "entry_1.png"))
        self.output_bg_1 = self.canvas.create_image(
            1492.44287109375,
            585.0015258789062,
            image=self.output_image_1
        )
        self.output_text = tk.Text(
            bd=0,
            bg="#1B2956",
            fg="white",
            highlightthickness=0
        )
        self.output_text.place(
            x=1367.94287109375,
            y=446.00152587890625,
            width=249.0,
            height=276.0
        )

        # Load and place graph button
        self.graph_image_2 = tk.PhotoImage(file=os.path.join('assets', 'main gui', "button_2.png"))
        self.graph_bt = tk.Button(
            image=self.graph_image_2,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: self.db_functions.generate_bar_graph(), # Calls the generate_bar_graph function when clicked
            activebackground='#0B194A',
            activeforeground='#0B194A',
            relief="flat"
        )
        self.graph_bt.place(
            x=1411.94287109375,
            y=317.00152587890625,
            width=162.0,
            height=63.0
        )

        # Creates multiple line UI elements
        self.canvas.create_rectangle(
            1400.94287109375,
            302.00152587890625,
            1401.94287109375,
            392.00152587890625,
            fill="#1B2956",
            outline="")

        self.canvas.create_rectangle(
            1400.94287109375,
            303.00152587890625,
            1584.9449310302734,
            304.00152587890625,
            fill="#1B2956",
            outline="")

        self.canvas.create_rectangle(
            1583.94287109375,
            302.00152587890625,
            1584.94287109375,
            392.00152587890625,
            fill="#1B2956",
            outline="")

        self.canvas.create_rectangle(
            1399.94287109375,
            391.00152587890625,
            1584.94287109375,
            392.00152587890625,
            fill="#1B2956",
            outline="")

        # Create and place the entry field for the last name
        self.last_name_image_2 = tk.PhotoImage(file=os.path.join('assets', 'main gui', "entry_2.png"))
        self.last_name_bg_2 = tk.Label(
            self.window,
            borderwidth=0,
            highlightthickness=0,
            image=self.last_name_image_2
        )
        self.last_name_bg_2.place(x = 1391.94287109375, y = 157.00152587890625,)

        self.last_name = tk.Entry(
            bd=0,
            bg="#1B2956",
            fg="white",
            insertbackground="white",
            font=("Bahnschrift SemiBold SemiConden", 10),
            highlightthickness=0
        )
        self.last_name.place(
            x=1394.0,
            y=160.0,
            width=200.0,
            height=40.0
        )

        # Create and place the entry field for the first name
        self.first_name_image_3 = tk.PhotoImage(file=os.path.join('assets', 'main gui', "entry_3.png"))
        self.first_name_bg_3 = tk.Label(
            self.window,
            borderwidth=0,
            highlightthickness=0,
            image=self.first_name_image_3
        )
        self.first_name_bg_3.place(x = 1391.94287109375, y = 75.00152587890625)

        self.first_name = tk.Entry(
            bd=0,
            bg="#1B2956",
            fg="white",
            insertbackground="white",
            font=("Bahnschrift SemiBold SemiConden", 10),
            highlightthickness=0
        )
        self.first_name.place(
            x=1394.0,
            y=78.0,
            width=200.0,
            height=40.0
        )

        # Places text for "First Name" and "Last Name"
        self.canvas.create_text(
            1391.94287109375,
            52.00152587890625,
            anchor="nw",
            text="First Name",
            fill="#5672A1",
            font=("Battambang Regular", 13 * -1)
        )

        self.canvas.create_text(
            1391.94287109375,
            134.00152587890625,
            anchor="nw",
            text="Last Name",
            fill="#5672A1",
            font=("Battambang Regular", 13 * -1)
        )

        # Load and place the search button
        self.search_image_3 = tk.PhotoImage(file=os.path.join('assets', 'main gui', "button_3.png"))
        self.search_bt = tk.Button(
            image=self.search_image_3,
            borderwidth=0,
            highlightthickness=0,
            # Calls the search_student function when clicked which takes the input in the first_name and last_name entry fields as parameters
            command=lambda: self.db_functions.search_student(self.first_name.get(), self.last_name.get()), 
            activebackground='#0B194A',
            activeforeground='#0B194A',
            relief="flat"
        )
        self.search_bt.place(
            x=1428.94287109375,
            y=224.00152587890625,
            width=129.0,
            height=36.54581069946289
        )

        self.canvas.create_rectangle(
            1380.94287109375,
            51.00154113769531,
            1381.94287109375,
            279.0213623046875,
            fill="#1B2956",
            outline="")

        self.canvas.create_rectangle(
            1379.94287109375,
            51.00152587890619,
            1603.94287109375,
            52.00152587890625,
            fill="#1B2956",
            outline="")

        self.canvas.create_rectangle(
            1602.94287109375,
            51.00152587890625,
            1603.94287109375,
            279.00372314453125,
            fill="#1B2956",
            outline="")

        self.canvas.create_rectangle(
            1379.94287109375,
            278.00152587890625,
            1603.94287109375,
            279.00152587890625,
            fill="#1B2956",
            outline="")

        # Load and place the start button
        self.start_image_4 = tk.PhotoImage(file=os.path.join('assets', 'main gui', "button_4.png"))
        self.start_bt = tk.Button(
            image=self.start_image_4,
            borderwidth=0,
            highlightthickness=0,
            command=self.start_face_recognition, # Calls the start_face_recognition method when clicked which begins face_recognition
            activebackground='#0B194A',
            activeforeground='#0B194A',
            relief="flat"
        )
        self.start_bt.place(
            x=483.0067138671875,
            y=786.0,
            width=162.0,
            height=63.0
        )

        # Load and place the stop button
        self.stop_image_5 = tk.PhotoImage(file=os.path.join('assets', 'main gui', "button_5.png"))
        self.stop_bt = tk.Button(
            image=self.stop_image_5,
            borderwidth=0,
            highlightthickness=0,
            command=self.stop, # Calls the stop method which stops face recognition
            activebackground='#0B194A',
            activeforeground='#0B194A',
            relief="flat"
        )
        self.stop_bt.place(
            x=692.0067138671875,
            y=786.0,
            width=162.0,
            height=63.0
        )

        # Creates multiple line UI elements
        self.canvas.create_rectangle(
            452.0067138671875,
            773.0,
            877.0067138671875,
            774.0,
            fill="#1B2956",
            outline="")

        self.canvas.create_rectangle(
            876.0067138671875,
            773.0,
            877.0067138828913,
            864.0,
            fill="#1B2956",
            outline="")

        self.canvas.create_rectangle(
            452.0067138671875,
            772.0,
            453.0067138671875,
            865.0,
            fill="#1B2956",
            outline="")

        self.canvas.create_rectangle(
            452.0,
            863.75,
            877.0080261230469,
            864.75,
            fill="#1B2956",
            outline="")

        # Initialises video capture
        self.video_source = 0
        self.vid = cv2.VideoCapture(self.video_source, cv2.CAP_DSHOW)
        self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) # Sets the camera frame width to 1280
        self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) # Sets teh camera frame height to 720
