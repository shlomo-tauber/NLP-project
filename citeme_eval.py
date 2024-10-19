from datasets import load_dataset
from RAG import CitationFinder

def dataset(split):
    ds = load_dataset("bethgelab/CiteME")
    # filter rows from "train" by feature "split"
    ds = ds['train'].filter(lambda x: x['split'] == split)
    return ds


def eval(split):
    ds = dataset(split)
    agent = CitationFinder()
    for i, row in enumerate(ds):
        if i > 0:
            break
        print(row)
        print(agent.find_citation(row['excerpt']))

if __name__ == '__main__':
    # argparse
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='test')
    args = parser.parse_args()

    eval(args.split)