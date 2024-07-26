# Facial Module Kick Off

These python files were developed as part of a collaboration with Mooveo project. It was focused on the kick off of the 
facial aspect of the project. During this period of time, due to the stage where the main project was on, most of the 
codes are pointed towards feature extraction rather than the interpretation of said results. 

In this repository, files corresponding to analysis of several facial areas and functions to analyze:

- Inferior section
  * Chin Amplitude
  * Chin Tremor
  * Nasolabial Fold
- Superior section
  * Blink Characterization
  * Eyebrow movement and amplitude
  * Correspondence between eyelid movements and eyebrows
- Facial Entropy Measurement
  * Forehead entropy
  * Cheek entropy
    * Left
    * Right
    * Joint
  * Chin Entropy

All of the scripts corresponding to those functions will be explained in detail in this very guide. For any questions 
that may arise during the usage of said codes, do not hesitate to contact me. 

## Inferior Facial Section

For this task, the main focus was to analyze chin tremor since, despite the database provided during the development of 
these codes did not have those many cases that presented this particular characteristic, the literature described this
very issue. Apart from that, the ampitude of the mouth during talking and resting stage and of the nasolabial folds can
be observed.

### Coords_chin.py

This file is formatted into argparse, that is, it can be easily used by inputting through terminal. The only modifications 
needed are the keypoints desired, which currently are set to measure the amplitude and the tremor of 4 keypoints that are 
lined up in the chin with a reference point located in the upper lip. 

This code will output a json that has, for each frame, a dictionary where the coordinates of the reference point and the
coordinates corresponding to the previously mentioned keypoints. 

It has associated another file (**_chin_distances.py_**), corresponding to the distance calculation which will also ask 
for the json file of the coordinates. 

### Coords_nasolabial.py

This file is also set to be used with argparse. It is expected to be input just a video since the keypoints are set to be
just that purpose. It is coded with the purpose of comparison by distances (**_nasolabial_distances.py_**) point to 
point of each side of the nasolabial fold.

It is intended to measure the rigidity of this particular area which is expected to present a higher rigidity, which will
translate into flatter graphs, or higher mobility, corresponding to more chaotic data. 

## Superior Facial Section

In this section, the objective was the analysis of eye movement, brow movement and forehead activity. Forehead rigidity
is believed to be an important feature for Parkinson's diagnosis, for that, plottable points are calculated to visualize
and analyze that specific characteristic. 

### Coords_blink.py

In this script, we can find a similar structure as nasolabial coordinates extraction. The main difference with it is its
capability of differentiating between left and right. With that characteristics, with just one video qas an input, two 
different json files will be output, each corresponding to a different side that will be used as independent signals.

To deal with these two at the same time, its associated distance calculation (_**blink_distances.py**_) takes care of it with ease, since the file 
asks for the general name that it has, that is:

When the file is first executed, it will ask for a video and as file name. If the file name is said to be "_blink.json_",
then the output will be "_blink_left.json_" and "_blink_right.json_". When calculating the distances, the input should be
"_blink.json_", since the code will look for both left and right. 


### Coords_brow.py

The structural thought behind this file is essentially the same as the one designed for the blink analysis. It will ask 
for a filename and will separate the characteristics in two separate signals, differentiating between left and right. 

It is currently set to use as a reference point the keypoint 168 which is the middle point between both brows. It can be
changed easily. 

It also has associated its corresponding distance calculation (_**brow_distances.py**_) that will consider that reference 
point and calculate all of the distances in each frame with that point as the "stable" one. These signals could be used
to analyze the amplitude of the eyebrow movement, or the general movement of the forehead, since if the signal comes out
as flat, then the forehead movement is expected to be lower, that is, Parkinson's patient.

### Peaks_finding.py

This aims to characterize both signals (brow and blink) simultaneously since, during the visualization of the signals, a
biometric patter was found, where everytime a blink was due, an slight movement of the brows was found.

Since that corresponding brow movement is subtle, a local maxima is found in the neighbouring frames where a peak in the
blink signal is detected. 

With this script, the following characterizations can be calculated for the final module:

- Amplitude of the blink
- Frequency of the blink
- Maximum size of the corresponding brow movement
- Activity of the eyebrow signal

## Facial Entropy

This codes measures the facial entropy in different regions of the face. It is mainly focused on forehead, cheeks, both
left and right, and chin. With it, it is expected to measure the movements that each subject has.

When the entropy is low, the patient is assumed to have Parkinson's disease when the entropy of each area is low, since
the movements corresponding to the tracking is not chaotic, hence, not high entropy. Similarly, a control is expected to
be found when the entropy is higher, since unexpected movements during a monologue are due when no facial rigidity is 
found. 

It also works with argparse and the only input is the video. It will print out the entropy values. 
