# MPRA-BPNet

----
## Step 0:
config.py: contains all the models. Needed by almost all the scripts
create_one_hot_seq.py: create one hot encoded sequences for all the following analyses

----
## Step 1: Train the new top layer (regression) and get the prediction. 
The output file is named "cv_free_MPRA.csv" bacause the prediction was not made using cross_val_predict but predicted using the entire "neural_network,bpnet_bottleneck_feat" as input and the top layer as the model. 
Snakefile_new_top_layer (requires util_cross_val.py)

----
## Step 2: Reconstruct the model by combining the top layer with the bottleneck model. 
Snakefile_reconstruction (requires util_cross_val.py and reconstruction_util.py)

----
## Step 3:
plot.py (to get spearman's correlation)

----
## Step 4: Run DeepLIFT and tfmodisco on the reconstructed models
Snakefile_modisco 
	I made some modifications to plot_weights() in modisco/visualization/viz_sequence.py:
		specifically, I added a name parameter, which is the name of the image generated, and plt.show() was changed to plt.savefig(name) to save the image.
modisco_pattern_generator.py
