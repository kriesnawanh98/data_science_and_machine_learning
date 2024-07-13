import nbformat

try:
    # Read the notebook
    with open('spaceship_eda.ipynb', 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    # Write the notebook back to file (this can fix some issues)
    with open('spaceship_eda_repaired.ipynb', 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    
    print("Notebook repaired successfully.")
except Exception as e:
    print(f"Failed to repair the notebook: {e}")
