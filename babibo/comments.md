# Descripttion 
Scripts for Babibo experiment. Each experiment consists of four files: two videos and two csvs. The pairs can be considered merged, but the final aois should be produced for each video separately. 

# Data processing methods
* Make a merged pool of AOIs from two example files, mark with 1 if aoi was present on the right of the screen (see csv).
* Using white screen detection, find starts and ends of the videos, mark them with corresponding item from the csv -> event files.
* Generate AOIs by rearranging the AOI pool using events
    * move aoi to the left or to the right depending on the csv. 

