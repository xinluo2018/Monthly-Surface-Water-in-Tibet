#! /bin/bath
## des: some example for fast processing of the remote sensing image.

##### shell style
### ------ downsampling ------ 
gdal_translate -outsize 20% 20% -r average -co COMPRESS=LZW $path_in $path_out 
## the -tr value georeferenced units, degree or meter
gdal_translate -tr 1000 1000 -r average -co COMPRESS=LZW $path_in $path_out 

### ------ mosaic ------ 
gdal_merge.py -init 0 -co COMPRESS=LZW -o $path_out $paths_in

### ------ subset ------ 
# extent: str(ulx) str(uly) str(lrx) str(lry)
gdal_translate -projwin -co COMPRESS=LZW $extent $path_in $path_out  


##### python style
```python
import os
## extent: str(ulx) str(uly) str(lrx) str(lry)
command = 'gdal_translate -projwin ' + extent +' -co COMPRESS=LZW ' + path_wat + ' ' + path_wat_subs
print(os.popen(command).read())
```
