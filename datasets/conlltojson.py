from argparse import ArgumentParser
import json

def process(file_in, file_out):
    out = []
    with open(file_in) as fh:
        fh.readline() # Skip header line

        words = []
        ner = []

        for line in fh:
            line = line.strip()
            if '\t' not in line:
                out.append(json.dumps({
                    'words': words,
                    'ner': ner
                })+'\n')
                words = []
                ner = []
                continue

            token, label = line.split('\t')
            words.append(token)
            ner.append(label)
            
    
    with open(file_out, 'w') as fh:
        fh.writelines(out)
        

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", help="Input conll file")
    parser.add_argument("-o", "--output", help="Output csv file")

    args = parser.parse_args()
    process(args.input, args.output)
