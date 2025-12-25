import sys
sys.path.insert(0, "/Users/ido.faran/personal/vis-tutorial")

import os
os.chdir("/Users/ido.faran/personal/vis-tutorial")

from evaluate_enhanced import main
import sys

sys.argv = [
    "evaluate_enhanced.py",
    "--checkpoint", "checkpoints/nightly_second_2025-12-24_06-57-15/best_model.pth",
    "--phase", "2"
]

if __name__ == "__main__":
    main()

