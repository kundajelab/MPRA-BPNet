"""Generate the motif pattern from the hdf5 files generated from modisco"""
from __future__ import print_function, division

import modisco.cluster.core
import modisco.cluster.phenograph.cluster
import modisco.cluster.phenograph.core
import modisco.affinitymat.core
from matplotlib import pyplot as plt
from modisco.visualization import viz_sequence
from collections import Counter
import modisco.util
import modisco.metaclusterers
import modisco.coordproducers
import modisco.core
import modisco.value_provider
import modisco.cluster
import modisco.aggregator
import modisco.tfmodisco_workflow.workflow
import modisco.tfmodisco_workflow.seqlets_to_patterns
import modisco.affinitymat
import modisco.nearest_neighbors
import modisco.backend
from config import model_exps
try:
    reload  # Python 2.7
except NameError:
    try:
        from importlib import reload  # Python 3.4+
    except ImportError:
        from imp import reload

import h5py
import numpy as np
import modisco
reload(modisco)
reload(modisco.backend.tensorflow_backend)
reload(modisco.backend)
reload(modisco.nearest_neighbors)
reload(modisco.affinitymat.core)
reload(modisco.affinitymat.transformers)
reload(modisco.tfmodisco_workflow.seqlets_to_patterns)
reload(modisco.tfmodisco_workflow.workflow)
reload(modisco.aggregator)
reload(modisco.cluster.core)
reload(modisco.cluster.phenograph.core)
reload(modisco.cluster.phenograph.cluster)
reload(modisco.value_provider)
reload(modisco.core)
reload(modisco.coordproducers)
reload(modisco.metaclusterers)
reload(modisco.util)


reload(viz_sequence)

reload(modisco.affinitymat.core)
reload(modisco.cluster.phenograph.core)
reload(modisco.cluster.phenograph.cluster)
reload(modisco.cluster.core)
reload(modisco.aggregator)

experiments = list(model_exps.values())
datasets = ['synthetic', 'genomic']
data_hparams = 'pool_size=15'
model = 'LinearRegressor'

for exp in experiments:
    for dataset in datasets:
        tfmodisc_saved = '0604_modisco/reconstruction/'+exp+'/MPRA/' + \
            dataset+'/pool_size=15/LinearRegressor/tfmodisco_results.hdf5'
        figure_save = '0604_modisco/reconstruction/'+exp + \
            '/MPRA/' + dataset+'/pool_size=15/LinearRegressor/'
        try:
            hdf5_results = h5py.File(tfmodisc_saved, "r")
        except:
            print(tfmodisc_saved)
            continue
        import seaborn as sns
        activity_patterns = np.array(hdf5_results['metaclustering_results']['attribute_vectors'])[
            np.array(
                [x[0] for x in sorted(
                    enumerate(
                        hdf5_results['metaclustering_results']['metacluster_indices']),
                    key=lambda x: x[1])])]
        plt.close()
        sns.heatmap(activity_patterns, center=0)
        plt.savefig(figure_save + 'metaclustering_results.png')

        metacluster_names = [
            x.decode("utf-8") for x in
            list(hdf5_results["metaclustering_results"]
                 ["all_metacluster_names"][:])]

        all_patterns = []

        for metacluster_name in metacluster_names:
            print(metacluster_name)
            metacluster_grp = (hdf5_results["metacluster_idx_to_submetacluster_results"]
                                           [metacluster_name])
            print("activity pattern:", metacluster_grp["activity_pattern"][:])
            all_pattern_names = [x.decode("utf-8") for x in
                                 list(metacluster_grp["seqlets_to_patterns_result"]
                                                     ["patterns"]["all_pattern_names"][:])]
            if (len(all_pattern_names) == 0):
                print("No motifs found for this activity pattern")
            for pattern_name in all_pattern_names:
                print(metacluster_name, pattern_name)
                all_patterns.append((metacluster_name, pattern_name))
                pattern = metacluster_grp["seqlets_to_patterns_result"]["patterns"][pattern_name]
                print("total seqlets:", len(
                    pattern["seqlets_and_alnmts"]["seqlets"]))
                background = np.array([0.27, 0.23, 0.23, 0.27])
                print("Task 0 hypothetical scores:")
                viz_sequence.plot_weights(
                    pattern["task_hypothetical_contribs"]["fwd"], figure_save + "hypothetical_scores"+pattern_name)
                print("Task 0 actual importance scores:")
                viz_sequence.plot_weights(
                    pattern["task_contrib_scores"]["fwd"], figure_save + "actual_importance_scores"+pattern_name)
                print("onehot, fwd and rev:")
                viz_sequence.plot_weights(viz_sequence.ic_scale(np.array(pattern["sequence"]["fwd"]),
                                                                background=background), figure_save + "fwd"+pattern_name)
                viz_sequence.plot_weights(viz_sequence.ic_scale(np.array(pattern["sequence"]["rev"]),
                                                                background=background), figure_save + "rev"+pattern_name)

        hdf5_results.close()
