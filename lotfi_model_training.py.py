import customtkinter as ctk;
from tkinter
import messagebox;

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib


class Page4(ctk.CTkFrame):
    """
    Page 4 - Model Training

    Features:
    1- Select target column
    2- Train machine learning model
    3- Show model accuracy
    4- Save trained model
    """

    def __init__(self, parent, dataframe):
        super().__init__(parent)

        self.dataframe = dataframe
        self.trained_model = None

        self.pack(fill="both", expand=True)
        self.build_ui()

    def build_ui(self):
        title = ctk.CTkLabel(
            self,
            text="Model Training",
            font=("Arial", 24, "bold")
        )
        title.pack(pady=20)

        self.target_input = ctk.CTkEntry(
            self,
            width=300,
            placeholder_text="Enter target column name"
        )
        self.target_input.pack(pady=15)

        ctk.CTkButton(
            self,
            text="Train Model",
            command=self.train_model
        ).pack(pady=10)

        ctk.CTkButton(
            self,
            text="Save Model",
            command=self.save_model
        ).pack(pady=10)

    def train_model(self):
        try:
            target_column = self.target_input.get().strip()

            if not target_column:
                messagebox.showwarning("Warning", "Enter target column name")
                return

            if target_column not in self.dataframe.columns:
                messagebox.showerror("Error", "Target column not found")
                return

            X = self.dataframe.drop(columns=[target_column])
            y = self.dataframe[target_column]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            self.trained_model = model

            accuracy = model.score(X_test, y_test)

            messagebox.showinfo(
                "Training Completed",
                f"Accuracy: {accuracy:.2f}"
            )

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def save_model(self):
        try:
            if self.trained_model is None:
                messagebox.showerror("Error", "Train model first")
                return

            joblib.dump(self.trained_model, "trained_model.pkl")

            messagebox.showinfo("Success", "Model saved successfully")

        except Exception as e:
            messagebox.showerror("Error", str(e))