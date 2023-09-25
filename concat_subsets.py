import json
from  glob import  glob
from pathlib  import  Path
dir_path = Path("data/Salesforce/codegen-16B-multi/THUDM/humaneval-x/greedy") # Replace with your path
split = "test" # Replace with your split
files = glob(str(dir_path) + f"/**/*.{split}.jsonl", recursive=True)


files.sort()
print(files)

json_list = []
with open(dir_path / (split + ".jsonl"), "w") as outfile:
    for fname in files:
        with open(fname) as infile:
            
            tag = fname.split("/")[-3]
            subset = fname.split("/")[-2]

            meta = {"subset": subset}
            
            for line in infile:
                data = json.loads(line)

                id = data["id"]
                part = data["part"]

                prompt = data["prompt"]
                reference = data["reference"]
                prediction = data["prediction"]
                ended = data["ended"]
                res = {
                    "id": id,
                    "part": part,
                    "prompt": prompt,
                    "reference": reference,
                    "prediction": prediction,
                    "ended": ended,
                    "meta": meta
                }
                line = json.dumps(res)
                outfile.write(line + "\n")
