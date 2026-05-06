#!/bin/bash
# Apply all code fixes needed to run the GNBR project
# Run this after cloning the repo: bash apply_fixes.sh

set -e

echo "=== Applying GNBR fixes ==="

# Fix 1: AdamW import in main.py (removed from transformers v5)
sed -i 's/from transformers import (/from torch.optim import AdamW\nfrom transformers import (/' main.py
sed -i '/^    AdamW,$/d' main.py
echo "[1/4] Fixed AdamW import in main.py"

# Fix 2: Test path in main.py (neuroblastoma.tsv -> test.tsv)
sed -i "s|'./data/neuroblastoma.tsv'|'./data/test.tsv'|" main.py
echo "[2/4] Fixed test path in main.py"

# Fix 3: Triple unpacking in dataloader.py (4 elements -> 3)
sed -i 's/                h, r, t, _ = triple/                h, r, t = triple/' dataloader.py
echo "[3/5] Fixed triple unpacking in dataloader.py"

# Fix 4: Add tqdm progress bar to training loop in trainer.py
sed -i "s/            for i_b, batch in enumerate(data_sampler):/            for i_b, batch in tqdm(enumerate(data_sampler), total=n_batch, desc=f'Epoch {epc}'):/" trainer.py
echo "[4/5] Added batch progress bar in trainer.py"

# Fix 5: Add outer epoch progress bar
sed -i "s/        for epc in range(self.load_epoch + 1, epoch):/        for epc in tqdm(range(self.load_epoch + 1, epoch), desc='Total Epochs', unit='epoch'):/" trainer.py
echo "[5/5] Added epoch progress bar in trainer.py"

echo ""
echo "=== All fixes applied! ==="
