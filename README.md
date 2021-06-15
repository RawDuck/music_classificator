## TEMAT PROJEKTU
 Klasyfikacja gatunku utworu na podstawie właściwości akustycznych.

## CEL PROJEKTU

Przygotowanie zbioru uczącego, wyekstrahowanie wybranych właściwości akustycznych. Docelowy program napisany w języku Python będzie przyjmował dowolny utwór w postaci pliku .wav, a następnie zwróci jeden z gatunków muzyki zdefiniowanych przez zbiór uczący. Dodatkowo zostanie przygotowany zbiór testowy, na którego podstawie zostanie określona dokładność klasyfikacji.

## BADANIA

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

## ZESPÓŁ
1. Łukasz Smalec
2. Robert Piątek
