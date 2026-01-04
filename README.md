# fraud-ml — Setup

Quick steps to fix the "Import could not be resolved" errors you see in the editor.

1) Create a venv (PowerShell)

```powershell
cd <path-to-this-folder>  # e.g. c:\Users\PC\Desktop\Backend\fraud-ml
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

2) (Optional) If your CSV files live outside this folder, set DATA_DIR before running:

PowerShell (temporary for the session):

```powershell
$env:DATA_DIR = 'C:\full\path\to\csv_folder'
python project_code.py
```

3) VS Code: make sure you select the interpreter at `.venv\Scripts\python.exe` for Pylance to find installed packages.

If you still see "Import could not be resolved" after installing, reload the window or re-open the folder in VS Code so the Python extension picks up the venv.
