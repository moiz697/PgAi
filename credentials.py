import tkinter as tk
from tkinter import ttk

def save_to_env():
    # Collecting the form data
    data = {
        'DB_HOST': host_entry.get(),
        'DB_PORT': port_entry.get(),
        'DB_NAME': dbname_entry.get(),
        'DB_USER': user_entry.get(),
        'DB_PASSWORD': password_entry.get(),
    }

    # Writing data to .env file
    with open('.env', 'w') as f:
        for key, value in data.items():
            f.write(f"{key}={value}\n")
    
    # Display a simple dialog box as feedback
    tk.messagebox.showinfo(title="Success", message="Database details saved successfully!")

# Setting up the GUI
root = tk.Tk()
root.title("PostgreSQL Connection Details")

# Creating form labels and entries
tk.Label(root, text="Host:").grid(row=0, column=0, padx=10, pady=5)
host_entry = ttk.Entry(root)
host_entry.grid(row=0, column=1, padx=10, pady=5)

tk.Label(root, text="Port:").grid(row=1, column=0, padx=10, pady=5)
port_entry = ttk.Entry(root)
port_entry.grid(row=1, column=1, padx=10, pady=5)

tk.Label(root, text="Database Name:").grid(row=2, column=0, padx=10, pady=5)
dbname_entry = ttk.Entry(root)
dbname_entry.grid(row=2, column=1, padx=10, pady=5)

tk.Label(root, text="User:").grid(row=3, column=0, padx=10, pady=5)
user_entry = ttk.Entry(root)
user_entry.grid(row=3, column=1, padx=10, pady=5)

tk.Label(root, text="Password:").grid(row=4, column=0, padx=10, pady=5)
password_entry = ttk.Entry(root, show="*")
password_entry.grid(row=4, column=1, padx=10, pady=5)

# Adding a button to trigger the save action
submit_btn = ttk.Button(root, text="Save", command=save_to_env)
submit_btn.grid(row=5, column=0, columnspan=2, pady=10)

root.mainloop()
