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

### Update: 5x5 training up and running in train_5x5.ipynb
The function 'bars' in classifierskxk.py has some crazy statements that honestly I'm super proud of.

The 5x5 eyeball is still really bad but seems slightly better than the 3x3 after very minimal training and experimentation. When training on 250 images it got to a steady 12% accurate classifications which is notably better than random 💪. 

The critical variables that effect the outcome of the 5x5 training are:
- **num_sections**, in kmeans_segmenter: this is the number of clusters the space of possible output vectors can get sectioned into. Now if you're a normal person (congrats) it would make sense for this to = 11 (one for each number 0-9, and one for other), but I happen to be a dreamer (delusional) and think that this eyeball has potential to classify far more than just numbers. Theoretically, num_sections can be as large as 2<sup>336</sup> (where 336 is the output space, assuming the output vector is binary which I'm now realizing it isn't but close enough) in which case it could classify literally anything. The downside of starting with large num_sections is that if the default output vector (probably just zeros) happens to randomly fall into a region above 10 then the model will never train bc it will never happen to be correct which it needs to do in order to start learning.
- **num_train**: this is the number of unique images the model will train on.
- **num_iters**: this is basically the number of training epochs, which multiplied with num_train gives you the total training steps.
- **alpha**, in new_kmeans_step: this is how much the amplification vector (don't get me started on the amplification vector) gets moved towards the center of the cluster that the output vector was correctly classified as (don't worry about it, tbh).
- **the demoninator in the else statement of that function**: this incredibly obvious and important variable determines how much to discount activated sensors cells in the amplification vector when the output is *incorrectly* classified. Currently set to 1 because nothing was happening and I was getting frustrated...
