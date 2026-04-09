"""Strip absolute paths from notebook outputs for docs publishing."""

import json
import re
import sys

path = sys.argv[1]
with open(path) as f:
    nb = json.load(f)
for cell in nb["cells"]:
    for out in cell.get("outputs", []):
        if "text" in out:
            out["text"] = [
                re.sub(r"/(?:Users|home|tmp|var)[^\s\"]*/", "", l) for l in out["text"]
            ]
with open(path, "w") as f:
    json.dump(nb, f, indent=1)
