# PAIP2019: Liver Cancer Segmentation
 
### Task 1: Liver Cancer Segmentation
### Task 2: Viable Tumor Burden Estimation
### This competition is part of the MICCAI 2019 Grand Challenge for Pathology.
***
### "utils.py" has been released for the participants who raised issues

It is a sample code for
1) loading masks
2) generating an overlay between an original image and a mask
3) resizing to make an image fit into simple viewers (e.g. Windows default viewer, Fiji, etc.)

Please note that this code is not providing svs loading part because there are several well-known open source library for this. (e.g. openslide, pyvips, etc.)

***

### "submission_compress.py" has been released for avoiding logistic issues in Grand-challenge platform
##### (If you don't compress each tif, the submission system may fail to score your results.)

It is another extremely simple code for
1) loading a mask
2) saving after applying ADOBE_DEFLATE level 9 compression with the same filename (may overwrite)
3) the output will have around ten times smaller file size compared to the original uncompressed tif

