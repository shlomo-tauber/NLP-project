from datasets import load_dataset



def eval(split):
    ds = load_dataset("bethgelab/CiteME")
    # filter rows from "train" by feature "split"
    ds = ds['train'].filter(lambda x: x['split'] == split)
    for i, row in enumerate(ds):
        if i > 10:
            break
        print(row)

if __name__ == '__main__':
    # argparse
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='test')
    args = parser.parse_args()
    eval(args.split)