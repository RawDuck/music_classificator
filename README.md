## MAIN IDEA
 Classification of the genre of a song based on its acoustic properties.

## GOAL

Preparation of the training set, extraction of selected acoustic properties. The target Python program will take any song as a .wav file and then return one of the genres of music defined by the training set. Additionally, a test set will be prepared, on the basis of which the classification accuracy will be determined.

## RESEARCH

1. SVM Accuracy with mfcc feature: 72.73%

| Genre |   %   |
| ----- | ----- |
| Rock  | 81.25 |
| Hiphop| 48.72 |
| Pop   | 87.10 |
| Metal | 80.00 |

Elapsed time: 25.990763425827026s

2. SVM Accuracy with fft feature: 75.76%

| Genre |   %   |
| ----- | ----- |
| Rock  | 81.25 |
| Hiphop| 61.54 |
| Pop   | 80.65 |
| Metal | 83.33 |

Elapsed time: 139.31962156295776s

3. SVM Accuracy with spectralCentroid feature: 36.36%

| Genre |   %   |
| ----- | ----- |
| Rock  | 43.75 |
| Hiphop| 12.82 |
| Pop   | 32.26 |
| Metal | 63.33 |

Elapsed time: 23.794891834259033s

