import argparse
import json


def sci_to_ltx(sci: str):
    return f'${sci.replace("e", "^{").replace("0", "")}}}$'

def frmt(f: float):
    # if f < .1:
    #     return sci_to_ltx('%.2e' % f)
    return '%.4f' % f


if __name__ == '__main__':
    # & ??? & ??? & ??? & ??? & ??? & ??? & ??? & ??? & ??? & ??? \\
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory", help="Base directory", type=str)
    parser.add_argument("-m", "--metrics",   help="List of metrics to output", type=str, default="eval_f1,eval_ece,f1_ece_ratio")
    parser.add_argument("-s", "--datasets",  help="List of datasets to walkthrough", type=str, default="conll2003,ncbi_disease,wikiann,GUM,wnut_17")
    
    args = parser.parse_args()

    line = ''
    for dataset in args.datasets.split(','):
        with open(f'{args.directory}/{dataset}/all_results.json') as fh:
            results = json.loads(fh.read())
        for metric in args.metrics.split(','):
            metric_value = ""
            if metric == "f1_ece_ratio":
                metric_value = frmt(results['eval_f1']**2/results['eval_ece'])
            else:
                metric_value = frmt(results[metric])
            line += f'& {metric_value} '

    line += '\\\\'

    print(line)

        
