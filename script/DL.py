import pandas as pd 
import os 
import numpy as np
import deepchem as dc
import tensorflow as tf
from itertools import product

def seed_all():
    np.random.seed(324)
    tf.random.set_seed(324)

seed_all()


def mt(lr,epoch):
    species = ['BS','RT','FHM','SHM']
    '''
    df = pd.read_csv('model_data.csv').replace({'LP':'BS'})
    df = df[df.remove.isna()]

    empty = df.drop_duplicates('cid')[['cid','Canonical_Smiles']].set_index('cid')
    
    for i in species:
        part = df[df.species==i].drop_duplicates('cid')
        part = part[['cid','pLD50','Canonical_Smiles']]
        part[i] = part['pLD50'].apply(lambda x:0 if x>-1 else 1)
        empty[i] = None
        empty.update(part.set_index('cid')[[i]])
    empty.to_csv(f'deepchem_dataloader/deepchem_MT.csv')
    '''
    loader = dc.data.CSVLoader(species,feature_field = 'Canonical_Smiles',featurizer=dc.feat.ConvMolFeaturizer(),id_field = 'cid')
    dataset = loader.create_dataset(f'deepchem_MT.csv')
    from deepchem import splits
    splitter = splits.RandomSplitter()
    train_set, test_set = splitter.train_test_split(dataset, frac_train=0.8)
    #test_set.to_csv('deepchem_dataloader/deepchem_MT_test.csv')
    model = dc.models.GraphConvModel(model_dir = f'GCN-MT-{lr}-{epoch}.m',n_tasks=len(species), mode='classification', batch_size=64,learning_rate=lr,dropout=0.5)
    #model.restore()
    model.fit(train_set,nb_epoch=epoch)
    d = {x:[] for x in ['model','AUC','ACC','recall','precision']}
    for idx,s in enumerate(species):
        auc = model.evaluate(test_set,dc.metrics.roc_auc_score,per_task_metrics=True)[1]['metric-1'][idx]
        acc = model.evaluate(test_set,dc.metrics.accuracy_score,per_task_metrics=True)[1]['metric-1'][idx]
        recall = model.evaluate(test_set,dc.metrics.recall_score,per_task_metrics=True)[1]['metric-1'][idx]
        f1 = model.evaluate(test_set,dc.metrics.f1_score,per_task_metrics=True)[1]['metric-1'][idx]
        print(auc,model.evaluate(test_set,dc.metrics.roc_auc_score,per_task_metrics=True))
        precision = 1/(2/f1-1/recall)
    
        d['model'].append(f'{s}_GCN_MT_{lr}_{epoch}')
        d['AUC'].append(auc)
        d['ACC'].append(acc)
        d['recall'].append(recall)
        d['precision'].append(precision)
    return d
store = pd.DataFrame()
for lr,epoch in product([0.001,0.005,0.01,0.05,0.1,0.5,1],[100,200,500]):
    store = pd.concat([store,pd.DataFrame(mt(lr,epoch)).round(3)])
    store.to_csv('ckpt.csv')