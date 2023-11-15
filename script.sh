#! /bin/bash

python single_shark.py best.pt assets/example_vid_1_trimmed.mp4 results/example_vid_1_trimmed_tracking.mp4 0.1
mv sample.json results/example_vid_1_trimmed_sample_data.json

python single_shark.py best.pt assets/example_vid_2.mp4 results/example_vid_2_tracking.mp4 0.1
mv sample.json results/example_vid_2_sample_data.json

python single_shark.py best.pt assets/example_vid_3.mp4 results/example_vid_3_tracking.mp4 0.1
mv sample.json results/example_vid_3_sample_data.json

# python single_shark.py best.pt assets/example_vid_4.mp4 results/example_vid_4_tracking.mp4 0.1
# mv sample.json results/example_vid_4_sample_data.json

python single_shark.py best.pt assets/example_vid_5.mp4 results/example_vid_5_tracking.mp4 0.1
mv sample.json results/example_vid_5_sample_data.json

python single_shark.py best.pt assets/example_vid_6_trimmed_and_zoomed.mp4 results/example_vid_6_trimmed_and_zoomed_tracking.mp4 0.1
mv sample.json results/example_vid_6_trimmed_and_zoomed_sample_data.json

