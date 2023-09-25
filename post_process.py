import json
from  glob import  glob
from pathlib  import  Path
dir_path = Path("data/Salesforce/codegen-16B-multi/THUDM/humaneval-x/greedy/python") # Replace with your path
split = "test" # Replace with your split
files = glob(str(dir_path / f"*.{split}.jsonl"))

files.sort()
print(files)

json_list = []
with open(dir_path / (split + ".jsonl"), "w") as outfile:
    for fname in files:
        with open(fname) as infile:
            for line in infile:
                data = json.loads(line)
                id = data["id"]
                part = data["part"]

                prompt = data["prompt"]
                reference = data["reference"]
                prediction = data["prediction"]
                ended = data["ended"]

                res = {"id": id, "part": part, "prompt": prompt, "reference": reference, "prediction": prediction, "ended": ended}
                line = json.dumps(data)
                outfile.write(line + "\n")
                #outfile.write(line)

data = []
with open(dir_path / (split + ".jsonl")) as file:
    for i, line in enumerate(file):
        data.append(json.loads(line))

    # sort by index and part
    data.sort(key=lambda x: (x["id"], x["part"][0]))

# save to file
with open(dir_path / (split + ".sorted.jsonl"), "w") as file:
    for i, line in enumerate(data):
        file.write(json.dumps(line) + "\n")


