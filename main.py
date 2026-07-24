

import tkinter as tk
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# ====================================================
# DATASET
# Symptoms:
# Fever, Cough, Headache, Fatigue
# ====================================================

data = [

    [1, 1, 0, 1, "Flu"],
    [1, 1, 1, 1, "Flu"],
    [1, 0, 1, 0, "Dengue"],
    [0, 0, 1, 1, "Migraine"],
    [0, 0, 0, 0, "Healthy"],
    [1, 1, 0, 0, "Cold"],
    [1, 0, 0, 1, "Viral Infection"],
    [0, 1, 1, 0, "Food Poisoning"],
    [1, 0, 0, 0, "Flu"],
    [0, 1, 0, 1, "Cold"],
    [1, 1, 1, 0, "Flu"],
    [0, 0, 1, 0, "Migraine"]

]


X = [row[:-1] for row in data]
y = [row[-1] for row in data]


# ====================================================
# TRAINING MACHINE LEARNING MODEL
# ====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)


model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


model.fit(
    X_train,
    y_train
)


# Model Accuracy

test_prediction = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    test_prediction
)


# ====================================================
# PREDICTION FUNCTION
# ====================================================

def predict_disease():

    symptoms = np.array([[
        fever_var.get(),
        cough_var.get(),
        headache_var.get(),
        fatigue_var.get()
    ]])

    result = model.predict(symptoms)[0]

    result_text.config(
        text=f"Predicted Disease:\n\n"
        f"{result}\n\n"
        f"Model Accuracy:\n"
        f"{accuracy*100:.2f}%"
    )


def clear():

    fever_var.set(0)
    cough_var.set(0)
    headache_var.set(0)
    fatigue_var.set(0)

    result_text.config(
        text="Result will appear here"
    )


# ====================================================
# TKINTER GUI
# ====================================================
window = tk.Tk()

window.title(
    "AI Disease Prediction System"
)

window.geometry(
    "550x700"
)

window.resizable(
    False,
    False
)


# ====================================================
# TITLE SECTION
# ====================================================

title_frame = tk.Frame(
    window,
    bg="#1f6aa5",
    height=100
)

title_frame.pack(
    fill="x"
)


title = tk.Label(
    title_frame,
    text="AI Disease Prediction System",
    font=("Arial", 20, "bold"),
    fg="white",
    bg="#1f6aa5"
)

title.pack(
    pady=15
)


subtitle = tk.Label(
    title_frame,
    text="Machine Learning Based Healthcare Assistant",
    font=("Arial", 11),
    fg="white",
    bg="#1f6aa5"
)

subtitle.pack()


# ====================================================
# INSTRUCTIONS
# ====================================================

instruction = tk.Label(
    window,
    text="Select symptoms and click Predict Disease",
    font=("Arial", 13)
)

instruction.pack(
    pady=20
)


# ====================================================
# SYMPTOM SELECTION
# ====================================================

symptom_frame = tk.LabelFrame(
    window,
    text="Symptoms",
    font=("Arial", 12, "bold"),
    padx=25,
    pady=15
)


symptom_frame.pack(
    padx=40,
    fill="x"
)


fever_var = tk.IntVar()
cough_var = tk.IntVar()
headache_var = tk.IntVar()
fatigue_var = tk.IntVar()


tk.Checkbutton(
    symptom_frame,
    text="Fever",
    variable=fever_var,
    font=("Arial", 12)
).pack(
    anchor="w"
)


tk.Checkbutton(
    symptom_frame,
    text="Cough",
    variable=cough_var,
    font=("Arial", 12)
).pack(
    anchor="w"
)


tk.Checkbutton(
    symptom_frame,
    text="Headache",
    variable=headache_var,
    font=("Arial", 12)
).pack(
    anchor="w"
)


tk.Checkbutton(
    symptom_frame,
    text="Fatigue",
    variable=fatigue_var,
    font=("Arial", 12)
).pack(
    anchor="w"
)


# ====================================================
# BUTTONS
# ====================================================

button_frame = tk.Frame(window)

button_frame.pack(
    pady=25
)


predict_button = tk.Button(
    button_frame,
    text="Predict Disease",
    command=predict_disease,
    bg="#1f6aa5",
    fg="white",
    font=("Arial", 12, "bold"),
    width=18,
    height=2
)


predict_button.grid(
    row=0,
    column=0,
    padx=10
)


clear_button = tk.Button(
    button_frame,
    text="Clear",
    command=clear,
    font=("Arial", 12),
    width=10,
    height=2
)


clear_button.grid(
    row=0,
    column=1
)


# ====================================================
# RESULT SECTION
# ====================================================

result_frame = tk.LabelFrame(
    window,
    text="Prediction Result",
    font=("Arial", 12, "bold"),
    padx=20,
    pady=15
)


result_frame.pack(
    padx=40,
    pady=10,
    fill="x"
)


result_text = tk.Label(
    result_frame,
    text="Result will appear here",
    font=("Arial", 14, "bold"),
    justify="center",
    height=5
)


result_text.pack(
    pady=10
)


# ====================================================
# FOOTER
# ====================================================

footer = tk.Label(
    window,
    text="Developed using Python + Random Forest + Tkinter",
    font=("Arial", 9),
    fg="gray"
)


footer.pack(
    pady=10
)


window.mainloop()
