from pathlib import Path
import seaborn as sns
import pandas as pd

import pickle
from matplotlib import pyplot as plt

def get_data(paths):
    results = {}
    for path in paths:
        domain = path.parent.stem
        if domain not in results:
            results[domain] = {}
        with open(path, 'rb') as fp:
            data = pickle.load(fp)
        for d in data:
            name = d['object_name']
            success = d['success']
            if name in results[domain]:
                results[domain][name]['trial'] += 1
                if success:
                    results[domain][name]['success'] +=1
            else:
                success = 1 if d['success'] else 0
                results[domain][name] = {'trial': 1,
                                        'success': success}
    return results
        # obj

def parse_data():
    ours = Path('/tmp/docker/result/ours').rglob('*.pkl')
   
    ours_result = get_data(ours)
    print(ours_result.keys())
    sr = []
    domain = []
    trials = []
    success = []
    obj = []
    model = []
    domain_map ={
        'duktig': 'Sink',
        'tight_cab': 'Cabinet',
        'high_cab': 'Top of shelf',
        'flat': 'Table',
        'suitcase': 'Suitcase',
        'basket': 'Basket',
        'grill': 'Grill',
        'drawer': 'Drawer',
        'circular_bin': 'Circular bin'
    }

    for k, v in ours_result.items():
        for kk, vv in v.items():
            domain.append(domain_map[k])
            trial = vv['trial']
            suc = vv['success']
            obj.append(kk)
            model.append('UNICORN-HAMNET')
            success.append(suc)
            trials.append(trial)
            sr.append(suc/trial)

    data = {
        'sr':sr,
        'object':obj,
        'num_trials':trials,
        'num_success': success,
        'model': model,
        'domain': domain
    }
    with open('/tmp/bench_result.pkl', 'wb') as fp:
        pickle.dump(data, fp)
    df = pd.DataFrame(data={
        'Success rate':sr,
        'object':obj,
        'num_trials':trials,
        'num_success': success,
        
        'model': model,
        'Domain': domain
    })
    print(df)
    sns.set_style("whitegrid")
    plt.figure(figsize=(11.5, 8.5))
    ax = sns.barplot(df,
                x='Domain',
                y='Success rate',
                hue='model',
                errorbar='sd',
                capsize=0.2,
                )
    ax.legend(prop={'size': 16})
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_ylabel('Success rate',fontsize=16)
    ax.set_xlabel('Domain',fontsize=16)
    plt.tight_layout()
    plt.savefig('/tmp/docker/bench_result.pdf')

if __name__ == '__main__':
    parse_data()