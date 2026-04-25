import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class VisualizationPage(tk.Frame):

    def __init__(self, parent, dataframe: pd.DataFrame, **kwargs):
        super().__init__(parent, **kwargs)

        # Store the DataFrame
        self.data = dataframe

        # Detect column types automatically
        self.numerical_cols = list(self.data.select_dtypes(include="number").columns)
        self.categorical_cols = list(self.data.select_dtypes(exclude="number").columns)
        self.all_cols = list(self.data.columns)

        # Build the UI
        self._build_ui()

    def _build_ui(self):

        # Title
        title = tk.Label(
            self,
            text="Data Visualization",
            font=("Helvetica", 16, "bold")
        )
        title.pack(pady=(16, 8))

        #Controls Frame
        controls = tk.Frame(self, padx=16, pady=8)
        controls.pack(fill="x")

        # Plot Type
        tk.Label(controls, text="Plot Type:", font=("Helvetica", 11)).grid(
            row=0, column=0, sticky="w", padx=8, pady=6
        )
        self.plot_type_var = tk.StringVar(value="Line")
        plot_type_menu = ttk.Combobox(
            controls,
            textvariable=self.plot_type_var,
            values=["Line", "Scatter", "Box"],
            state="readonly",
            width=18
        )
        plot_type_menu.grid(row=0, column=1, sticky="w", padx=8, pady=6)

        # X Column
        tk.Label(controls, text="X Column:", font=("Helvetica", 11)).grid(
            row=1, column=0, sticky="w", padx=8, pady=6
        )
        self.x_col_var = tk.StringVar(value=self.all_cols[0] if self.all_cols else "")
        x_col_menu = ttk.Combobox(
            controls,
            textvariable=self.x_col_var,
            values=self.all_cols,
            state="readonly",
            width=18
        )
        x_col_menu.grid(row=1, column=1, sticky="w", padx=8, pady=6)

        #  Y Column (numerical only)
        tk.Label(controls, text="Y Column:", font=("Helvetica", 11)).grid(
            row=2, column=0, sticky="w", padx=8, pady=6
        )
        self.y_col_var = tk.StringVar(
            value=self.numerical_cols[0] if self.numerical_cols else ""
        )
        y_col_menu = ttk.Combobox(
            controls,
            textvariable=self.y_col_var,
            values=self.numerical_cols,   # Y must always be numerical
            state="readonly",
            width=18
        )
        y_col_menu.grid(row=2, column=1, sticky="w", padx=8, pady=6)

        # Plot Button
        plot_btn = tk.Button(
            self,
            text="Plot",
            font=("Helvetica", 12, "bold"),
            bg="#4A90D9",
            fg="white",
            padx=20,
            pady=6,
            cursor="hand2",
            command=self._on_plot
        )
        plot_btn.pack(pady=16)


    def _validate(self, x_col: str, y_col: str, plot_type: str) -> str | None:

        if not x_col:
            return "Please select an X column."
        if not y_col:
            return "Please select a Y column."

        # Box plot: X should be categorical (or at least not the same as Y)
        if plot_type == "Box":
            if x_col not in self.categorical_cols:
                return (
                    f"For a Box plot, X should be a categorical column.\n"
                    f"Detected categorical columns: {self.categorical_cols or 'none'}"
                )

        # Line / Scatter: X should be numerical
        if plot_type in ("Line", "Scatter"):
            if x_col not in self.numerical_cols:
                return (
                    f"For a {plot_type} plot, X should be a numerical column.\n"
                    f"Detected numerical columns: {self.numerical_cols or 'none'}"
                )

        if x_col == y_col:
            return "X and Y columns must be different."

        return None


    def _on_plot(self):
        """Called when the user clicks 'Plot'. Validates then draws the chart."""
        x_col = self.x_col_var.get()
        y_col = self.y_col_var.get()
        plot_type = self.plot_type_var.get()

        # Validate first
        error = self._validate(x_col, y_col, plot_type)
        if error:
            messagebox.showerror("Invalid Selection", error)
            return

        # Draw the selected plot in a new matplotlib window
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.suptitle(f"{plot_type} Plot  —  {x_col} vs {y_col}", fontsize=13)

        try:
            if plot_type == "Line":
                sns.lineplot(data=self.data, x=x_col, y=y_col, ax=ax)

            elif plot_type == "Scatter":
                sns.scatterplot(data=self.data, x=x_col, y=y_col, ax=ax)

            elif plot_type == "Box":
                sns.boxplot(data=self.data, x=x_col, y=y_col, ax=ax)

        except Exception as e:
            # Close the empty figure and show the error
            plt.close(fig)
            messagebox.showerror("Plot Error", f"Could not generate plot:\n{e}")
            return

        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        plt.tight_layout()
        plt.show()


