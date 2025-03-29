# Description
This repository contains tools and scripts for processing results of a number of psycho-linguistic experiments. 
Experiments included eye-tracking, decision logging, reaction time measumerements, and so on. 
Data was logged using PsychoPy software and saved in form of .csv and .xlsx files. 
Screen recording was also done along the entirety of the experiments. 

Most scripts cross-process the resulting table data with the videos and eye-tracking data, to:
- chunk eye-tracking data into categories, according to the stimuli presented to the subjects;
- filter eye-tracking data by applying areas-of-interest, also according to the stimuli.
- where needed, extract event data from videos, such as stimulus type, reaction time, answer, etc.

# Directory structure
* __src/eyeoi__: contains importable tools for some of the tasks listed below, no 'executable scripts';
* __production task__: scripts and tools to production task;
* __log_parsing__: scripts and tools related to parsing of the psychopy logs;
* __text_completion__: same, for text complection;
* __sentence picture matching__: same for sentence-picture matching;

File naming convention:
* files containing tools and function definitions are mostly called _gerunds_ or _nouns_, e.g `event_matcher.py`.
* files containing executable scripts are mostly called using _verbs_, e.g. `extract_phrases.py`.

Descriptions on how to run executable files can be found in the top description or in main() function.