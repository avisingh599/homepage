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

Mentor:[Prof. Ashutosh Saxena](http://cs.stanford.edu/~asaxena) and  [Ashesh Jain](http://www.asheshjain.org) <br>
At **Cornell University** in collaboration with **Stanford AI Lab** <br>
[arXiv pre-print](http://arxiv.org/abs/1509.05016) | [website](http://www.brain4cars.com) <br> <br>
The project attempts to anticipate driving maneuvers using information from driver-facing and road-facing cameras. My contributions to the project are:

* **Revamped the Computer Vision pipeline** of the project by incorporating a Constrained Local Neural Field model for enhanced head pose tracking of the driver. This made the tracking robust to changes in illumination and pose, while providing consistent optical flow trajectories of facial landmarks and rich head pose features. 
* Improved the training algorithm for Autoregressive Input Output HMMs by initializing them with Gaussian Mixture Models, and using LBFG-S optimization algorithm for minimizing the cost function. 
* I carried out extensive experiments on the Brain4Cars dataset that improved the performance from a Precision/Recall of **77.4/71.2 to 86.7/78.2**. Further experimenting on Recurrent Neural Networks with Long Short Term Memory (LSTM) units boosted the performance to a Precision/Recall of **90.5/87.4**.
* Co-authored a [paper](http://arxiv.org/abs/1509.05016) which is now under review at ICRA 2016. 

![Improved face tracking](/images/icra2016/feat_compare_3.jpg)  |  ![Anticipatory Recurrent Neural Net architecture](/images/icra2016/fusionRNN.png)

**The project in popular press**: 


<div id="images">

<a href="http://fortune.com/2015/09/18/artificial-intelligence-cars/" target="_blank">
	<img  HEIGHT="30"  class="img-responsive" src="/images/fortune.png" alt="" />
</a>

<a href="http://www.forbes.com/sites/dougnewcomb/2015/04/22/new-technology-looks-inward-to-prevents-accidents-by-reading-drivers-body-language/" target="_blank">  
	<img HEIGHT="30"  class="img-responsive" src="/images/forbes.jpg" />
</a>

<a href="http://fusion.net/story/119737/the-car-that-knows-when-youll-get-in-an-accident-before-you-do/" target="_blank">  
    <img HEIGHT="30"  class="img-responsive" src="/images/fusion.png" />
</a>

</div>

---


<h3 class="nomargin">Stereo and Monocular Visual Odometry</h3>
Mentor: [Prof. KS Venkatesh](http://home.iitk.ac.in/~venkats/)<br>
At Computer Vision Lab, Electrical Engineering, IIT-Kanpur <br>
[github-1](https://github.com/avisingh599/mono-vo) | [github-2](https://github.com/avisingh599/vo-howard2008)  <br> <br>
I implemented three different visual odometry systems ( two stereo and one monocular), and evaluated them on the KITTI Visual Odometry benchmark.

#### Stereo Visual Odometry

* Jiang2014: Model Based ICP: 3D points triangulated from stereo data, inliers detected via the use Iterative Closest Point Algorithm that uses a 1-DOF motion model for initial estimate. Efficient PnP Algorithm is then used on the selected inliers to obtain the final rotation and translation
* Howard2008: Inlier detection using an assumption of scene rigidity. Problem reduced to finding the maximum clique in a graph, solved using a heuristic. Levenberg-Marquardt used for minimizing the reprojection error on the selected inliers. Implementation available on [Github](https://github.com/avisingh599/vo-howard08).

#### Monocular Visual Odometry

* A simple approach utilizing Nister's Five Point Algorithm was implemented. 
* Implmentation available on [Github](https://github.com/avisingh599/mono-vo).

---

<h3 class="nomargin">Landmark Based Robot Localization using RGB-D Data</h3>

Mentor: [Arjun Bashin](https://www.linkedin.com/pub/arjun-bhasin/ba/86b/2a8)<br>
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
Mentor: [Prof. Vinay Namboodiri](https://sites.google.com/site/vinaynamboodiri/)<br>
(Group of two) Course project for CS679: Machine Learning for Computer Vision<br>
[presentation (PDF)]({{site.url}}/assets/activity-classification.pdf)

* Hidden Conditional Random Fields were used for human activity classification from RGBD sequences.
* Pose features were extracted from a skeleton detection pipeline, and underwent a novel normalization operation for reducing the effect of variance in body sizes of different subjects
* An accuracy of 71% was achieved on a reduced version of MSR Daily Activity 3D Dataset (6-class classification)

--- 

<h3 class="nomargin">DAAnT - Dental Automated Analysis Tool (Computer Vision for intraoral cameras) </h3>
Mentor: [Dr.Hyunsung Park](https://sites.google.com/site/hyunsung/home), Postdoc at Camera Culture Group, MIT Media Lab<br>
At the week-long RedX camp organised by [Camera Culture Group](http://cameraculture.media.mit.edu/), [MIT Media Lab](http://www.media.mit.edu/)<br><br>
We devloped Computer Vision algorithms for automated diagnosis of dental problems from images obtained from an intraoral camera. My contribution in the eight-member team was primarily in the following areas:

* **Stitching Images**: Used **Affine SIFT** to overcome the difficulties in feature matching due to changes in perspective, and then computed Homography with RANSAC to stitch the images.
* **Segmenting Every tooth**: Marker-controlled **watershed transform** was used to segment every tooth from the image. Both semi-automated and automated
approaches were implemented and compared.


---

<h3 class="nomargin">Scene Flow</h3>
Mentor: [Prof. KS Venkatesh](http://home.iitk.ac.in/~venkats/)<br>
At Computer Vision Lab, Electrical Engineering, IIT-Kanpur <br><br>
Scene Flow is the extension of the well-known Optical Flow to 3D data. 

* Formulated and implemented two approaches for estimation of Scene Flow from RGB-D data, one an extension of Lucas-Kanade to 3D, and the other an extension of Horn-Schunck.
* Captured RGBD Data from a **Microsoft Kinect** using **OpenNI** and **OpenCV** libraries, and did all the processing using C++ in OpenCV.

---

### Oher projects:

1. Predicting Facial Attractiveness using Computer Vision. [[Github]](https://github.com/avisingh599/face-rating) [[Blog Post]](http://www.learnopencv.com/computer-vision-for-predicting-facial-attractiveness/)
2. Hilbert Tranform in Verilog. [[Github]](https://github.com/avisingh599/verilog-hilbert-transform)
3. Arduino-based Cashless Campus system with biometric authentication. [[Github]](https://github.com/avisingh599/arduino-cashless-campus)[[Youtube]](https://www.youtube.com/embed/cq6SSzsLrPU)

