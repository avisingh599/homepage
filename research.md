---
layout: page
permalink: /research/
title: "Research"
modified: 2015-03-06 14:35
tags: []
image:
  feature: 
  credit: 
  creditlink: 
share: false
---

<h3 class="nomargin">Brain4Cars: Learning Temporal Models to Anticipate Driving Maneuvers </h3>

Guide:[Prof. Ashutosh Saxena](http://cs.stanford.edu/~asaxena) and  [Ashesh Jain](http://www.asheshjain.org) <br>
At **Cornell University** in collaboration with **Stanford AI Lab** <br>
[arxiv pre-print](http://arxiv.org/abs/1509.05016) | [project website](http://www.brain4cars.com) <br>

<!--![Improved face tracking](/images/icra2016/feat_compare_3.jpg)  |  ![Anticipatory Recurrent Neural Net architecture](/images/icra2016/fusionRNN.png)-->

Brain4Cars addresses the problem of anticipating driver maneuvers several seconds before they happen. It fuses the information from driver-facing and road-facing cameras with data from other sensors to make its predictions. These predictions can then be passed to driver assistance systems that can warn the driver if the maneuver is deemed to be dangerous. My contributions to the project are listed below:

* The KLT face tracker in the project was replaced with a facial landmark localization pipeline based on Constrained Local Neural Fields. This provided robust tracking and allowed the computation of head pose, which then served as a strong feature for maneuver anticipation.
* Implemented Gaussian Mixture Model-based initialization, and LBFGS optimization for training Autoregressive Input Output Hidden Markov Models (AIOHMM), which are a modification of HMMs and used for anticipation in Brain4Cars.
* The performance of the AIOHMM-based anticipation system improved from a Precision/Recall of **77.4/71.2 to 86.7/78.2**.
* Further testing on a Long Short Term Memory (LSTM) network (replacing the AIOHMM) increased the performance to a Precision/Recall of **90.5/87.4**.
* Co-authored a [paper](http://arxiv.org/abs/1509.05016) which is now under review at ICRA 2016.

|Feature Extraction  |  Anticipatory LSTM |
|--------------------------------------|------------------------------| 
| <img src="/images/icra2016/feat_compare_3.jpg" alt="alt text" width="" height="300"> | <img src="/images/icra2016/fusionRNN.png" alt="alt text" width="" height="290"> |


The project in popular press: 


<div id="images">

<a href="http://www.technologyreview.com/news/541866/this-car-knows-your-next-misstep-before-you-make-it/" target="_blank">
	<img  HEIGHT="30"  class="img-responsive" src="/images/mit.png" alt="" />
</a>

<a href="http://fortune.com/2015/09/18/artificial-intelligence-cars/" target="_blank">
	<img  HEIGHT="30"  class="img-responsive" src="/images/fortune.png" alt="" />
</a>

<a href="http://www.forbes.com/sites/dougnewcomb/2015/04/22/new-technology-looks-inward-to-prevents-accidents-by-reading-drivers-body-language/" target="_blank">  
	<img HEIGHT="30"  class="img-responsive" src="/images/forbes.jpg" />
</a>

</div>

---

<h3 class="nomargin">Deep Learning for Visual Question Answering </h3>

Guide:[Prof. Amitabh Mukherjee](http://www.cse.iitk.ac.in/users/amit/) <br>
At IIT-Kanpur as a course project <br>
[poster](http://home.iitk.ac.in/~avisingh/cs671/project/poster.pdf) | [code](https://github.com/avisingh599/visual-qa) | [blog](http://avisingh599.github.io/deeplearning/visual-qa/)<br>

![Sample Results](/images/vqa/sample_results.jpg) 

I built a system that can answer open-ended questions about images, using ConvNets, LSTMs and Word Embeddings as building blocks. I worked with [VQA](http://www.visualqa.org) dataset, which has over 600K natural language questions about real world images.

* The image is passed through a **Convolutional Neural Network** (pre-trained on the ImageNet dataset), and the activations from the last hidden layer are extracted. This 4096-dimensional vector serves as a fixed-length representation of the image. 
* Every word in the question is converted to its word vector using the Glove Word Embeddings, and these embeddings are sequentially passed to a **Long Short Term Memory Network** (LSTM). The output of the LSTM after all the words have been passed is used as an embedding for the question.
* The image embedding and the question embedding are concatenated and passed through two fully connected layers with 50% dropout. The entire network (except the CNN) is trained end-to-end. 
* The VQA dataset was used with about 370K questions for training and 240K questions for testing. The system was able to achieve an accuracy of **53.34%** on the test-dev split of the VQA dataset.

----


<h3 class="nomargin">Stereo and Monocular Visual Odometry</h3>
Guide: [Prof. KS Venkatesh](http://home.iitk.ac.in/~venkats/)<br>
At Computer Vision Lab, Electrical Engineering, IIT-Kanpur <br>
[code-1](https://github.com/avisingh599/mono-vo) | [code-2](https://github.com/avisingh599/vo-howard2008)  <br> <br>
I implemented three different visual odometry systems ( two stereo and one monocular), and evaluated them on the KITTI Visual Odometry benchmark.

#### Stereo Visual Odometry

* Jiang2014: Model Based ICP: 3D points triangulated from stereo data, inliers detected via the use Iterative Closest Point Algorithm that uses a 1-DOF motion model for initial estimate. Efficient PnP Algorithm is then used on the selected inliers to obtain the final rotation and translation
* Howard2008: Inlier detection using an assumption of scene rigidity. Problem reduced to finding the maximum clique in a graph, solved using a heuristic. Levenberg-Marquardt used for minimizing the reprojection error on the selected inliers. Implementation available on [Github](https://github.com/avisingh599/vo-howard08).

#### Monocular Visual Odometry

* A simple approach utilizing Nister's Five Point Algorithm was implemented. 
* Implmentation available on [Github](https://github.com/avisingh599/mono-vo).

---

<h3 class="nomargin">Landmark Based Robot Localization using RGB-D Data</h3>

Guide: [Arjun Bashin](https://www.linkedin.com/pub/arjun-bhasin/ba/86b/2a8)<br>
At Mechatronics Lab, Mechanical Engineering, IIT-Kanpur <br>

* Geometric Triangulation was used to determine the pose (modelled as a set of random variables with
Gaussian Distribution) of a robot, from information obtained from noisy landmarks.
* A Microsoft Kinect was used to identify the landmarks (using color histogram based models), CAMshift
algorithm was used to track these landmarks, and bearing measurements were calculated using the depth
data.
* An error model for the data obtained from a Kinect was used along with the Error Propagation Law
to arrive at the uncertainty in the final pose computed using the Geometric Localisation Algorithm.

---

<h3 class="nomargin">Hidden CRFs for Human Activity Classification from RGB-D sequences</h3>
Guide: [Prof. Vinay Namboodiri](https://sites.google.com/site/vinaynamboodiri/)<br>
(Group of two) Course project for CS679: Machine Learning for Computer Vision<br>
[presentation (PDF)]({{site.url}}/assets/activity-classification.pdf)

* Hidden Conditional Random Fields were used for human activity classification from RGBD sequences.
* Pose features were extracted from a skeleton detection pipeline, and underwent a novel normalization operation for reducing the effect of variance in body sizes of different subjects
* An accuracy of 71% was achieved on a reduced version of MSR Daily Activity 3D Dataset (6-class classification)

--- 

<h3 class="nomargin">DAAnT - Dental Automated Analysis Tool (Computer Vision for intraoral cameras) </h3>
Guide: [Dr.Hyunsung Park](https://sites.google.com/site/hyunsung/home), Postdoc at Camera Culture Group, MIT Media Lab<br>
At the week-long RedX camp organised by [Camera Culture Group](http://cameraculture.media.mit.edu/), [MIT Media Lab](http://www.media.mit.edu/)<br><br>
We devloped Computer Vision algorithms for automated diagnosis of dental problems from images obtained from an intraoral camera. My contribution in the eight-member team was primarily in the following areas:

* **Stitching Images**: Used **Affine SIFT** to overcome the difficulties in feature matching due to changes in perspective, and then computed Homography with RANSAC to stitch the images.
* **Segmenting Every tooth**: Marker-controlled **watershed transform** was used to segment every tooth from the image. Both semi-automated and automated
approaches were implemented and compared.


---

<h3 class="nomargin">Scene Flow</h3>
Guide: [Prof. KS Venkatesh](http://home.iitk.ac.in/~venkats/)<br>
At Computer Vision Lab, Electrical Engineering, IIT-Kanpur <br><br>
Scene Flow is the extension of the well-known Optical Flow to 3D data. 

* Formulated and implemented two approaches for estimation of Scene Flow from RGB-D data, one an extension of Lucas-Kanade to 3D, and the other an extension of Horn-Schunck.
* Captured RGBD Data from a **Microsoft Kinect** using **OpenNI** and **OpenCV** libraries, and did all the processing using C++ in OpenCV.

---

### Oher projects:

1. Predicting Facial Attractiveness using Computer Vision. [[Github]](https://github.com/avisingh599/face-rating) [[Blog Post]](http://www.learnopencv.com/computer-vision-for-predicting-facial-attractiveness/)
2. Hilbert Tranform in Verilog. [[Github]](https://github.com/avisingh599/verilog-hilbert-transform)
3. Arduino-based Cashless Campus system with biometric authentication. [[Github]](https://github.com/avisingh599/arduino-cashless-campus)[[Youtube]](https://www.youtube.com/embed/cq6SSzsLrPU)

