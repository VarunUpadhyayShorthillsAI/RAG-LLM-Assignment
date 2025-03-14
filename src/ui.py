import tkinter as tk
from tkinter import scrolledtext
from main1 import medical_query_input  # Import the function from main.py

def on_submit():
    """Handles the submit button click."""
    query = entry.get()  # Get the user's query
    if not query:
        output_area.insert(tk.END, "Please enter a query.\n")
        return

    # Clear the output area
    output_area.delete(1.0, tk.END)

    # Call the medical_query_input function
    try:
        response = medical_query_input(query)  # Get the response
        # Extract only the generated answer (remove headers and processing message)
        if "=== Generated Answer ===" in response:
            answer_start = response.find("=== Generated Answer ===") + len("=== Generated Answer ===")
            answer_end = response.find("=== Supporting Context ===")
            answer = response[answer_start:answer_end].strip()
        else:
            answer = response.strip()
        
        output_area.insert(tk.END, answer)  # Display only the answer
    except Exception as e:
        output_area.insert(tk.END, f"Error: {str(e)}\n")
# Create the main window
root = tk.Tk()
root.title("Medical Query System")

# Create and place the input field
entry_label = tk.Label(root, text="Enter your medical query:")
entry_label.pack(pady=5)

entry = tk.Entry(root, width=80)
entry.pack(pady=5)

# Create and place the submit button
submit_button = tk.Button(root, text="Submit", command=on_submit)
submit_button.pack(pady=10)

# Create and place the output area
output_area = scrolledtext.ScrolledText(root, width=100, height=20)
output_area.pack(pady=10)

# Run the application
root.mainloop()