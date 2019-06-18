model_exps = {
    "nexus/binary.gw": 'nexus,gw,OSNK,1,0,0,FALSE,same,0.5,64,25,0.001,9,FALSE',
    "nexus/binary+profile.gw": 'nexus,gw,OSNK,1,0.1,0.01,FALSE,same,0.5,64,25,0.001,9,FALSE',
    "nexus/profile.gw": "nexus,gw,OSNK,0,10,1,FALSE,same,0.5,64,25,0.001,9,FALSE",
    "nexus/profile.peaks.bias-corrected": 'nexus,peaks,OSNK,0,10,1,FALSE,same,0.5,64,25,0.004,9,FALSE,[1,50],TRUE',
    "nexus/profile.peaks.bias-corrected.augm": 'nexus,peaks,OSNK,0,10,1,FALSE,same,0.5,64,25,0.004,9,FALSE,[1,50],TRUE,TRUE',
    "nexus/profile.peaks.non-bias-corrected": 'nexus,peaks,OSNK,0,10,1,FALSE,same,0.5,64,25,0.004,9,FALSE-2',
    "seq/binary.gw": 'seq,gw,OSN,1,0,0,FALSE,same,0.5,64,50,0.001,9,FALSE',
    "seq/profile.peaks.bias-corrected": 'seq,peaks,OSN,0,10,1,FALSE,same,0.5,64,50,0.004,9,FALSE,[1,50],TRUE',
    "seq/profile.peaks.bias-corrected.augm": 'seq,peaks,OSN,0,10,1,FALSE,same,0.5,64,50,0.004,9,FALSE,[1,50],TRUE,TRUE',
    "seq/profile.peaks.non-bias-corrected": 'seq,peaks,OSN,0,10,1,FALSE,same,0.5,64,50,0.004,9,FALSE',
    'nexus/profile.peaks-union.bias-corrected': 'nexus,nexus-seq-union,OSN,0,10,1,FALSE,same,0.5,64,25,0.004,9,FALSE,[1,50],TRUE',
    'seq/profile.peaks-union.bias-corrected': 'seq,nexus-seq-union,OSN,0,10,1,FALSE,same,0.5,64,50,0.004,9,FALSE,[1,50],TRUE',
}
