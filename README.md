# BANKNOTE UNIQUE CODE RECOGNITION

<img height=300 src="assets/roi.png"></img>

## OVERVIEW

The application is supposed to capture region of interest(**ROI**) with unique banknote id, which consists of two letters and seven numbers(0-9), and implement **OCR** on it in order to recognize id, so it would be possible to operate with this code in future.

In this project Russian currency is used as an example of work(one, two and five thousands of rubles).

> **IT'S IMPORTANT** Project has not been COMPLETED yet. At the moment app isn't working correctly, only 67% of planned functions are included. The app doesn't have properly working ocr.

### PROBLEM PROJECT SOLVES

The main problem that this application solves is the theft of banknotes. If your money has been stolen and you have the id of each bill, then you can provide these numbers to the police, which will greatly increase the chance of returning your lost money to your pocket.

## APP ARCHITECTURE

### CREATING OBJECT DETECTION MODEL TO CAPTURE REGION OF INTEREST

First it's required to detect roi, because OCR needs to see only the text we're interested in without any other distracting characters and signs. Bills are type of things that doesn't have a lot of different patterns(1000 of rubles looks like the other million of 1000 ruble bill and so forth). So, it would be really difficult to teach model on such little pattern as text(our id), especially if we resize banknote pics to the size of 170x128. 

And it was decided to teach the model to detect whole banknote and slice this region the way to get ~1/4 of right up part of banknote(where id is located).

> Check pic below, where blue region is object's annotation and red is the piece with id, obtained by cropping the image.

<img height=300 src="assets/roi_example.png"></img>

#### ANNOTATON PROCESS

label-studio was used to annotate the samples. Out of 904 available samples only 860 were suitable.

#### MODEL ARCHITECTURE

VGG-16 is used as foundation to experiment with. Activation function of output layer was changed from softmax to sigmoid, also last layer contains 4 neurons, because model will predict 4 metrics: top left x, top left y, down right x, down right y. Input shape was changed to 170x128 instead of 224x224

Deep learning part second's dense layer neurons amount was fine-tuned to 4000. Categorical cross-entropy loss function was changed to MSE, because we have regression task now, not classification.

<img height=300 src="assets/vgg16_architecture.png"></img>

<details>
  <summary>
  Model's code
  </summary>
  add code here in future
</details>

