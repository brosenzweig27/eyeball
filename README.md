# Eyeball
The eyeball<sup>(TM)</sup> is an absolutely terrible method for the pretty basic computer vision task of classifying handwritten numbers from the [MNIST data set](https://www.kaggle.com/competitions/digit-recognizer/data?select=train.csv).

## Premise
From what I remember from the two neuroscience classes I took in undergrad, human vision works bc we have a bunch of cells / clusters of cells that are activated by very specific types of inputs. For example, certain cells directly downstream of receptor cells in the retina activate only when dark bar of a specific orientation is passed over the field of vision (source: I vaguely remember this from undergrad). I thought this would be an interesting jumping-off point for a new species of computer vision models that essentially are comprised of a bunch of individual sensors that serve one specific function.

Version 0 of this model (found in numdet.ipynb) analyzes the 28x28 pixel greyscale images of handwritten numbers of the MNIST training set (found in digit-recognizer/train.csv) through a collection of 14 distinct sensor types. Each distint 3x3 region of the image is assigned each of the following 14 types of sensors, resulting in a length 9464 binary vector (1 if the corresponding sensor is activated, 0 if not):
1. Empty
2. Full
3. Dark vertical bar on the left
4. Light vertical bar on the right
5. Light vertical bar in the middle
6. Dark vertical bar in the middle
7. Light vertical bar on the left
8. Dark vertical bar on the right
9. Dark horizontal bar on the top
10. Light horizontal bar on the bottom
11. Light horizontal bar in the middle
12. Dark horizontal bar in the middle
13. Light horizontal bar on the top
14. Dark horizontal bar on the bottom

I am too lazy to write the rest of this ReadMe so understanding the rest of this code is left as an exercise to the reader. It should be noted that the 3x3 sensors alone do an incredibly terrible job of classifying the numbers but I'm hopeful, largely bc I'm delusional. I'm working on adding 5x5 sensors rn so stay tuned...
