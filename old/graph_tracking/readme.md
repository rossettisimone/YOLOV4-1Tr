Please look at the images for a momentary explanation.
The output is in idx_objs and has size (num_frames x max_num_bbs)
The first row is just the labels of each bb, which in principle corresponds to frame0. 
The second up to the last_but_one row, indicates the bounding boxes association, corresponding to the label in the first row.

[![Output](https://github.com/fiora0/trackers/blob/rot3d/CompareFrames/XSimon/Figure_33.png)](#idx_objs).

For example, suppose  we are at frame 9 then the tracker finds that the bounding box labeled by 0 continues to be labeled by 0 in frame 10, that the bounding box labeled by 1, which in frame 9 had no association, now is labeled by 1, and the bounding boxes labeled by 2  and 3 have no associations because their value is -1.

[![Track between F4-F5](https://github.com/fiora0/trackers/blob/rot3d/CompareFrames/XSimon/figureX.png)](#tr1),
[![Track between F29-F30](https://github.com/fiora0/trackers/blob/rot3d/CompareFrames/XSimon/Figure_5.png)](#features).

If we look at the  images we see that the yellow bounding box indicates that the current bb (eg. bb_5) is not identified in the previous image. The tiny bbs of few pixels are faked bounding boxes used to maintain the track for each label, though they are never associated to predicted bbounding boxes as they only generate noise.
