import fire
from pathlib import Path
import re

def merge(path1, path2, output_path):
    filepaths1 = Path(path1).glob('*.txt')
    prog = re.compile(r'^\d{6}.txt$')
    filepaths1 = filter(lambda f: prog.match(f.name), filepaths1)
    for fp1 in list(filepaths1):
        with open(fp1) as f1:
            contents = f1.readlines()
        if len(contents) != 0:
            contents += "\n"
        with open(Path(path2) / f"{fp1.stem}.txt", 'r') as f2:
            contents += f2.readlines()
        with open(Path(output_path) / f"{fp1.stem}.txt", 'w') as f:
            f.writelines(contents)

if __name__ == '__main__':
    fire.Fire()

