#!/bin/bash

cd /home/yons/Desktop/developer-luo/Monthly-Surface-Water-in-Tibet

# rm -r data/dset/valsite_wat_infer/gscales/as/model_?
# rm -r data/dset/valsite_wat_infer/gscales/des/model_?
# rm -r data/dset/valsite_wat_infer/gscales/as_des/model_?
# rm -r data/dset/valsite_wat_infer/scales/as_des/model_?
# rm -r data/dset/valsite_wat_infer/single/as_des/model_?

rm -r model/trained_model/gscales/traset/as/model_*
rm -r model/trained_model/gscales/traset/des/model_*
rm -r model/trained_model/gscales/traset/as_des/model_*
rm -r model/trained_model/scales/traset/as_des/model_*
rm -r model/trained_model/single/traset/as_des/model_*