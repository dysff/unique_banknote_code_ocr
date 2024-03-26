# BANKNOTE UNIQUE CODE RECOGNITION

<p align='center'>
<img height=300 src="assets/roi.png"></img>
</p>

<p align='center'>
  <img src="https://img.shields.io/badge/python%203.10.0-black?style=flat-square&color=blue">
  <img src="https://img.shields.io/badge/ROI-black?style=flat-square&color=red">
  <img src="https://img.shields.io/badge/OCR-black?style=flat-square&color=green">
  <img src="https://img.shields.io/badge/Object%20detection-black?style=flat-square&color=brown">
</p>

## TABLE OF CONTENTS

- [OVERVIEW](#overview)
  - [METHODS](#methods)
- [MODEL'S PERFORMANCE](#models-performance)
- [MODEL'S ARCHITECTURE](#models-architecture)
- [FIXED PROBLEMS](#problems-encountered-in-the-process)
- [GETTING STARTED](#getting-started)

## OVERVIEW

The application is supposed to capture region of interest(**ROI**) with unique banknote id, which consists of two letters and seven numbers(0-9), and implement **OCR** on it in order to recognize id, so it would be possible to operate with this code in future.

In this project Russian currency is used as an example of work(one, two and five thousands of rubles).

> [!IMPORTANT]
> Project has not been COMPLETED yet. At the moment app isn't working correctly, only 67% of planned functions are included. The app doesn't have properly working ocr.

### METHODS
<p>
    <ul>
      <li>Optical character recognition</li>
      <li>Object detection</li>
      <li>Data annotation</li>
      <li>Model's architecture fine-tuning</li>
      <li>Learning curves visualization</li>
      <li>Relative coordinates</li>
      <li>Image resize</li>
      <li>Augmentation implementation </li>
    </ul>
</p>

### PROBLEM PROJECT SOLVES

The main problem that this application solves is the theft of banknotes. If your money has been stolen and you have the id of each bill, then you can provide these numbers to the police, which will greatly increase the chance of returning your lost money to your pocket.

Application will be using by people through a telegram bot. Person sends picture of a banknote to telegram bot and after it saves id to the ids database. Each person has private database.

## APP ARCHITECTURE

### MODEL'S PERFORMANCE

At the moment OCR has 12.5% accuracy. Object detection model crops region of interest with accuracy of 98%. To improve OCR performance it should be fine-tuned on type of the font used on Russian banknotes. So, that's the problem is being solved now.

#### EXAMPLE OF WORKING

<p align='center'>
<img src="assets/example.png"></img>

#### LEARNING CURVES

<p align='center'>
<img height=400 width=400 src="assets/learning_curves.png"></img>
</p>

It's easy to see, that's model is working not quite well. Training loss is higher than test and there's huge gap between them. It indicates that the model is overfitting. **So, it's the problem is being solved now.**

With many tries and different architectures this is the best result this project ever had at the moment.

### CREATING OBJECT DETECTION MODEL TO CAPTURE REGION OF INTEREST

First it's required to detect roi, because OCR needs to see only the text we're interested in without any other distracting characters and signs. Bills are type of things that doesn't have a lot of different patterns(1000 of rubles looks like the other million of 1000 ruble bill and so forth). So, it would be really difficult to teach model on such little pattern as text(our id), especially if we resize banknote pics to the size of 170x128. 

And it was decided to teach the model to detect whole banknote and slice this region the way to get ~1/4 of right up part of banknote(where id is located).

> [!NOTE]
> Check pic below, where blue region is object's annotation and red is the piece with id, obtained by cropping the image.

<p align='center'>
<img height=300 src="assets/roi_example.png"></img>
</p>

#### ANNOTATON PROCESS

label-studio was used to annotate the samples. Out of 904 available samples only 860 were suitable.

#### MODEL'S ARCHITECTURE

VGG-16 is used as foundation to experiment with. Activation function of output layer was changed from softmax to sigmoid, also last layer contains 4 neurons, because model will predict 4 metrics: top left x, top left y, down right x, down right y. Input shape was changed.

Deep learning part was fine-tuned, so it has a bit different architecture. Categorical cross-entropy loss function was changed to MSE, because we have regression task now, not classification.

<p align='center'>
<img height=400 width=400 src="assets/vgg16_architecture.png"></img>
</p>

## PROBLEMS ENCOUNTERED IN THE PROCESS

### RELATIVE COORDINATES

There was no any idea how to work with relative coordinates. It was needed to transform absolute coords to relative, because the model is using sigmoid activation function, where it ranges from 0 to 1.

**It was solved by** dividing each coordinate by appropriate shape. Here's function, which transfrom these coordinates.

```python
def relative_coords(bbox):
  bbox = [bbox[0] / cols,#top_x
          bbox[1] / rows,#top_y
          bbox[2] / cols,#top_x
          bbox[3] / rows]#top_y
  
  return bbox
```

### NOT ENOUGH DATA

First versions of model were overfitting a lot more. All of that because there're were only 860 data samples.

**It was partly solved by** implementing augmentation. Best result of all augmentation types was shown by cut-out aug. It decreased overfitting in around two times.

## GETTING STARTED

The application does not work for other users, but for the owner. It uses pretrained regression convolutional neural network.

## License <a href="LICENSE"> ![GitHub](https://img.shields.io/github/license/MarketingPipeline/Awesome-Repo-Template) </a>

This project is licensed under the MIT License - see the
[LICENSE.md](LICENSE) file for
details.