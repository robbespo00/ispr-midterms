# ISPR-Midterms

# Midterm 1

Assignment 1

As you know, auto-regressive models assume weak stationarity. What happens if this assumption does not hold? To study the effect of non-stationarity, we will add a linear trend to the “Appliances” column of the dataset, which measures the energy consumption of appliances across a period of 4.5 months. 
- First, preprocess the dataset to remove any trend (if necessary)
- Perform an autoregressive analysis on the clean time series
- Add a linear trend to the time series
- Perform the autoregressive analysis on the new time series

Show the results of the analysis with and without the linear trend, discussing your design choices and the results.
To perform the autoregressive analysis, fit an autoregressive model on the first 3 months of data and estimate performance on the remaining 1.5 months. Remember to update the autoregressive model as you progress through the 1.5 testing months. For instance, if you have trained the model until time T, use it to predict at time T+1. Then to predict at time T+2 retrain the model using data until time T+1. And so on. You might also try and experimenting with less "computationally heavy" retraining schedule (e.g. retrain only "when necessary").   You can use the autoregressive model of your choice (AR, ARMA, ...). 

Hint: in Python, use the ARIMA class of the statsmodels library (set order=(3,0,0) for an AR of order 3); in Matlab you can use the ar function to fit the model and the forecast function to test.

Assignment 2

Plot the auto-correlogram of the temperature data in the appliances dataset (i.e. the columns marked as Ti). Make a plot for each of the sensors (try to put 4/5 of them in the same slide in a readable form). For this assignment, it is sufficient to show the plots and discuss any trends that are found from this analysis.

Assignment 3

The musical pitch of a note is determined by its fundamental frequency. The pitch played by different instruments sounds different due to harmonics, i.e. other frequencies that are superimposed and determine the timbre of the instrument. This dataset contains samples from several instruments playing different notes. Plot the spectrogram for some of them (4 instruments are sufficient) and check if it is possible to recognize the different instruments by only looking at the spectrogram. In your presentation, discuss which samples you chose to compare, how you computed the spectrogram and whether the resulting features are sufficient to recognize the instrument.

In Python you can import WAVs (and acces several other music-related functions), using the LibROSA library.
Image processing assignments

All the image processing assignments require to use the following dataset:

http://download.microsoft.com/download/A/1/1/A116CD80-5B79-407E-B5CE-3D5C6ED8B0D5/msrc_objcategimagedatabase_v1.zip

The dataset includes original images as well as their semantic segmentation in 9 object classes (i.e. the image files whose name ends  in “_GT”, where each pixel has a value which is the identifier of the semantic class associated to it).  Each file has a name starting with a number from 1 to 8, which indicates the thematic subset of the image, followed by the rest of the file name. This tematic subset can be used for instance as a class for the full image in image classification tasks.

Assignment 4
Perform image segmentation on all images in the dataset, using the normalized cut algorithm run on the top of superpixels rather than on raw pixels. For each image compute a performance metric (which one it is up to you to decide) measuring the overlap between the image segments identified by NCUT and the ground truth semantic segmentation. You do not need to show this metric for all images, rather focus on on selecting and discussing 2 examples of images that are well-segmented Vs 2 examples of images that are badly segmented (according to the above defined metric). 

Hint: in Python, you have an NCut implementation in the scikit-image library; in Matlab, you can use the original NCut implementation here. Superpixels are implemented both in Matlab as well as in OpenCV. Feel free to pickup the implementation you liked most (and motivate the choice).

Assignment 5

Select four thematic subsets of your choice, out of the total 8 available, and collect all the associated images. For these images, extract the SIFT descriptors using the visual feature detector embedded in SIFT to identify the points of interest. Aggregate all the identified descriptors in a dataset and run k-means (or any clustering algorithm of your choice) on such data to partition the descriptors in clusters. Then analyze the obtained clusters by confronting the descriptors assigned to each cluster with the thematic classes of the images from which they were extracted (in other words, compute a confusion matrix between the clusters and the four thematic images). Discuss your findings. Choice of the number of clusters and of the clustering algorithm is on you (and should be discussed in the report).

Assignment 6

Implement the convolution of a Laplacian of a Gaussian blob (LoG) detector with an image and apply it to 3-4 images of your choice from the dataset (possibly from different thematic classes). Do not use library functions for implementing the convolution or to generate the LoG filter. Implement your own and show the code (the interesting bits at least)! The function you implement should be able to run the LoG for different choices of the scale parameter, which is passed as an input argument. Show the results of your code on the 3-4 example images, for different choices of the scale parameter (sigma).

# Midterm 2

Assignment 1

Fit an Hidden Markov Model with Gaussian emissions to the data in DSET1: it is sufficient to focus on the “Appliances” and “Lights” columns of the dataset which measure the energy consumption of appliances and lights, respectively, across a period of 4.5 months. Consider the two columnsin isolation, i.e. train two separate HMM, one for appliances and one for light.  Experiment with HMMs with a varying number of hidden states (e.g. at least 2, 3 and 4). Once trained the HMMs, perform Viterbi on a reasonably sized subsequence (e.g. 1 month of data) and plot the timeseries data highlighting (e.g. with different colours) the hidden state assigned to each timepoint by the Viterbi algorithm.  Then, try sampling a sequence of at least 100 points from the trained HMMs and show it on a plot discussing similarities and differences w.r.t. the ground truth data.

Assignment 2

Implement a simple image understanding application for DSET2 using the LDA model and the bag of visual terms approach described in Lecture 12. For details on how to implement the approach see the BOW demo and paper [12] referenced on the Moodle site.  Keep one picture for each image subset (identified by the initial digit in the filename) out of training for testing. In short:

1.     For each image (train and test) extract the SIFT descriptors for the interest points identified by the MSER detector.

2.     Learn a 500-dimensional codebook (i.e. run k-means with k=500 clusters) from  the SIFT descriptors of the training images (you can choose a subsample of them if k-means takes too long).

3.     Generate the bag of visual terms for each image (train and test): use the bag of terms for the training images to train an LDA model (use one of the available implementations). Choose the number of topics as you wish (but reasonably).

4.     Test the trained LDA model on test images: plot (a selection of) them with overlaid visual patches coloured with different colours depending on the most likely topic predicted by LDA.

Assignment 3

Implement from scratch an RBM and apply it to DSET3. The RBM should be implemented fully by you (both CD-1 training and inference steps) but you are free to use library functions for the rest (e.g. image loading and management, etc.).

1.     Train an RBM with a number of hidden neurons selected by you (single layer) on the MNIST data (use the training set split provided by the website).

2.     Use the trained RBM to encode all the images using the corresponding activation of the hidden neurons.

3.     Train a simple classifier (e.g. any simple classifier in scikit) to recognize the MNIST digits using as inputs their encoding obtained at step 2. Use the standard training/test split. Show a performance metric of your choice in the presentation/handout.

Assignment 4 
Implement a Bayesian Network (BN) comprising at least 10 nodes, all with binomial or multinomial distribution. Represent the BN with the data structures that you deem appropriate and in the programming language that you prefer. The BN should model some problem/process of your choice, so you are also free to define the topology according to your prior knowledge (just be ready to justify your choices). For instance, you can define a BN to represent a COVID diagnosis through a certain number of events/exams/symptoms: e.g. Cough, Cold, Fever, Breathing problems, Swab Exam, etc. Or you can model your daily routine: Wakeup, Coffee, Toilet, Study, Lunch, etc.
Once you have modelled the BN, also plug in the necessary local conditional probability tables. You can set the values of the probabilities following your own intuition on the problem (ie no need to learn them from data). Then run some episoded of Ancestral Sampling on the BN and discuss the results.

The assignment needs to be fully implemented by you, without using BN libraries.

Assignment 5

Learn the structure of the Bayesian Network (BN) resulting from the dataset DSET4 using two BN structure learning algorithms of your choice.  For instance you can consider the algorithms implemented in PGMPY or any other comparable library (e.g. see the list of libraries listed in Lecture 7).  Compare and discuss the results obtained with the two different algorithms. Also discuss any hyperparameter/design choice you had to take.

# Midterm 3

Assignment 1

DATASET (MNIST): http://yann.lecun.com/exdb/mnist/

Train a denoising or a contractive autoencoder on the MNIST dataset: try out different architectures for the autoencoder, including a single layer autoencoder, a deep autoencoder with only layerwise pretraining and a deep autoencoder with fine tuning. It is up to you to decide how many neurons in each layer and how many layers you want in the deep autoencoder. Show an accuracy comparison between the different configurations.

given the encoding z1 of image x1 and z2 of image x2, a latent space interpolation is an encoding that obtained with the linear interpolation z* = a*z1 + (1 - a)*z2, with a in [0, 1]. Perform a latent space interpolation and visualize the results using:
- z1 and z2 from the same class
- z1 and z2 from different classes
Plot the results, for example by showing the image reconstructions for a=0.0, 0.1, 0.2, …, 1.0. Are the resulting images plausible digits?

Try out what happens if you feed one of the autoencoders with a random noise image and then you apply the iterative gradient ascent process described in the lecture to see if the reconstruction converges to the data manifold.

Assignment 2

DATASET (CIFAR-10): https://www.cs.toronto.edu/~kriz/cifar.html

Implement your own convolutional network, deciding how many layers, the type of layers and how they are interleaved, the type of pooling, the use of residual connections, etc. Discuss why you made each choice a provide performance results of your CNN on CIFAR-10.  

Now that your network is trained, you might try an adversarial attack to it. Try the simple Fast Gradient Sign method, generating one (or more) adversarial examples starting from one (or more) CIFAR-10 test images. It is up to you to decide if you want to implement the attack on your own or use one of the available libraries (e.g. foolbox,  CleverHans, ...). Display the original image, the adversarial noise and the final adversarial example.  

Assignment 3

DATASET (ENERGY PREDICTION): https://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction

Train two recurrent neural networks of your choice, one non-gated (Simple RNN) and one gated (LSTM, GRU) to predict the energy consumption one-step-ahead:

2.  Setup a one step-ahead predictor for energy expenditure, i.e. given the current energy consumption, predict its next value.
3. experiment with different uses of teacher forcing during training:
    1. using teacher forcing during the entire training
    2. using a schedule where you reduce the amount of teacher forcing over time
    3. using no teacher forcing 

Show and compare performance of both architectures and the different teacher forcing methods.

Assignment 4

DATASET (PRESIDENTIAL SPEECHES): https://www.kaggle.com/datasets/littleotter/united-states-presidential-speeches

Pick up one of the available implementations of the Char-RNN (e.g. implement1,  implement2,  implement3, implement4, etc.) and train it on the presidential speech corpora. In particular, be sure to train 2 separate models, one on all the speeches from President Clinton and  one on all the speeches from President Trump. Use the two models to generate new speeches and provide some samples of it at your choice. Should you want to perform any other analysis, you are free to do so.

Perform the following analyses:
- a “hidden state seed” is an hidden state that computed by feeding a sequence as input to the Char-RNN. The resulting hidden state is used as a the initial hidden state and can be used to control the generation. Try to use the following seeds:
    - a Clinton sentence fed to the Clinton model (and viceversa)
    - a Clinton sentence fed to the Trump model (and viceversa)
    - a random sentence (e.g. from Wikipedia)

Please note that the speech files contain XML tags: be sure to remove them before feeding the text to the Char-RNN (or you might consider leaving just the <APPLAUSE> and/or the <BOOING> tags to see if the network is smart enough to understand when the speech reaches a climax).  

Assignment 5

DATASET (LERCIO HEADLINES) - Dataset collected by Michele Cafagna

As in Assignment 4, pick one of the CHAR-RNN implementations and train one model on the dataset which contains about 6500 headlines from the Lercio satirical newspage, scraped by Michele Cafagna, past student of the ISPR course. The dataset is contained in a CSV file, one line per headlines. Be aware that the dataset can be a bit noisy (some errors due to encoding conversions) so you might need some preprocessing in order to prepare it for the task. Also, I am afraid the dataset is in Italian only as this is the language of the newspage.

Try experimenting with different configurations of the CHAR-RNN, varying the number of layers. Since the dataset is quite small, keep the number of hidden neurons contained otherwise the net will overfit. Use the trained model (the best or the worst, your choice) to generate new headlines.  

Perform the following analyses:
- The softmax has a temperature parameter T that you can use to control the randomness of the output distribution (i.e. output logits are divided by T). Experiment with different values of T and comment the results.

Assignment 6

DATASET (Fake News Classification) - Dataset

The dataset contains real and fake news, including their title, text, subject, and date. The objective is to train a binary classifier to recognize fake news. You are free to choose the model's architecture, but you should describe and justify your design choices. 

Notice that the fake and real news in the dataset are balanced. However, in the real world, real news are much more frequent than fake ones (hopefully). Simulate the effect of the data imbalance by undersampling/oversampling one of the classes in the training set and compute the test accuracy on a (balanced) test set. Then, try to use a mechanism to make the training robust to imbalances, such as weighting the loss for the samples depending on their class. Discuss the results of this mitigation.

# Midterm 4

List of papers
Peters & Schaal, Reinforcement learning of motor skills with policy gradients, Neural Networks, 2008
Gu et al. Q-Prop: sample-efficient policy gradient with an off-policy critic, ICLR, 2017
Schulman et al, Trust Region Policy Optimization, ICML, 2015
Auer et al, Finite-time Analysis of the Multiarmed Bandit Problem, Machine Learning, 2002
Stadie et al, Incentivizing Exploration In Reinforcement Learning With Deep Predictive Models, ICLR 2016
Rusu et al, Policy distillation, ICLR 2016
Levine and Koltun, Guided policy search, ICML 2013
Ho and Ermon, Generative Adversarial Imitation Learning, NIPS 2016
Ng and Russell, Algorithms for Inverse Reinforcement Learning, ICML 2000
Ziebart et al, Maximum Entropy Inverse Reinforcement Learning, AAAI 2008
Finn et al, Guided Cost Learning: Deep Inverse Optimal Control via Policy Optimization, ICML 2016
Hausman, Multi-Modal Imitation Learning from UnstructuredDemonstrations using Generative Adversarial Nets, NIPS 2017
Igl et al, Deep Variational Reinforcement Learning for POMDPs, ICML 2018
Karol Hausman, et al. Learning an Embedding Space for Transferable Robot Skills, ICLR 2018
Carlos Florensa, et al. Automatic Goal Generation for Reinforcement Learning Agents, ICML 2019
