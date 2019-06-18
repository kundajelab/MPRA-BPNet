"""Create one-hot encoded sequences from the input"""
import pandas as pd
from basepair.exp.chipnexus.data import(pool_bottleneck,
                                                 gen_padded_sequence,
                                                 syn_padded_sequence,
                                                 )

import numpy as np

input_gen = '../tidied_GEN_RPMsExpression_plusSeqs'
input_syn = '../tidied_SYN_RPMsExpression_plusSeqs'


from concise.preprocessing import encodeDNA
dfs_gen = pd.read_csv(input_gen)
dfs_syn = pd.read_csv(input_syn)

bpnet_seq_gen = encodeDNA([gen_padded_sequence(s, "AAAGACGCG")
                                   for s in dfs_gen.Sequence.str.upper()])


bpnet_seq_syn = encodeDNA([gen_padded_sequence(s, "AAAGACGCG")
                                   for s in dfs_syn.Sequence.str.upper()])


np.save("tidied_GEN_RPMsExpression_plusSeqs_one_hot",bpnet_seq_gen)
np.save("tidied_SYN_RPMsExpression_plusSeqs_one_hot",bpnet_seq_syn)
