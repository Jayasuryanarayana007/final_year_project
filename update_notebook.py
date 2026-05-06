import json

file_path = "c:/project0403/gnbr_kaggle_notebook.ipynb"

with open(file_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        src = "".join(cell.get("source", []))
        if "python main.py" in src and "--load_epoch 2" not in src:
            source = cell["source"]
            # modify last line
            source[-1] = source[-1] + " \\\n"
            source.append("    --load_epoch 2")
            
            # insert the checkpoint copying mechanism
            idx = 0
            for i, line in enumerate(source):
                if line.startswith("python main.py"):
                    idx = i
                    break
            
            source.insert(idx, "find /kaggle/input -name \"*history*.pkl\" -exec cp {} params/lp/ \\; || true\\n")
            source.insert(idx, "find /kaggle/input -name \"*metric_hits1.pt\" -exec cp {} params/lp/ \\; || true\\n")
            source.insert(idx, "echo 'Copying saved checkpoints for resumption...'\\n")
            break

with open(file_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Notebook updated successfully!")
