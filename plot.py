"""Calculate spearmans correlation between the prediction and the original values"""
from basepair.utils import read_pkl
from basepair.imports import *
paper_config()
df_mpra = pd.read_csv('output_0604_top_layer/reconstruction/reconstructed_MPRA.csv')
from config import model_exps
model_exps_inf = {v:k for k,v in model_exps.items()}
df_mpra['model'] = df_mpra.exp.map(model_exps_inf)
with pd.option_context("display.max_colwidth", 100):
    print(df_mpra.query('data=="genomic"')[['model', 'metrics/genomic/spearmanr']].
          sort_values("metrics/genomic/spearmanr").
          to_string())
with pd.option_context("display.max_colwidth", 100):
    print(df_mpra.query('data=="synthetic"')[['model', 'metrics/synthetic/spearmanr']].
          sort_values("metrics/synthetic/spearmanr").
          to_string())
#df_act = pd.read_csv('../../../src/chipnexus/train/seqmodel/output/activity.csv')
#df_act = df_act[df_act.split == 'test']
#df_act[[
        #'model_kwargs/bn', 
        #'model_kwargs/n_hidden/0',
        #'model_kwargs/pool_size',
        #'model_kwargs/pool_type']].drop_duplicates()
#with pd.option_context("display.max_colwidth", 100):
    #print(df_act[['model',
        #'model_kwargs/bn', 
        #'model_kwargs/pool_size',
        #'model_kwargs/pool_type', 'metrics/H3K27ac/spearmanr', 'metrics/PolII/spearmanr']].
          #sort_values("metrics/PolII/spearmanr").
          #to_string())
#df_act['model'] = df_act.exp.map(model_exps_inf)
#print(df_act)
#print(df_act.head())
