A classic dataset for evaluation of a computer vision network is the Fashion MNIST Dataset.
This dataset consists of 70,000 28X28 pixels of different clothing items split into a 60,000 unit training set and a 10,000 unit test set. There are equal numbers of all clothing items within these items. It looks like this:

 
##Benchmark Accuracy
The creators of this dataset have supplied accuracy scores for each of the machine learning models built into the popular sci-kit learn package. We can see that the competitive models perform at or around the 85% to 90% accuracy mark. This provides a good benchmark for evaluating our neural net.

#Building the Network

Our network is going to consist of 3 parts:
1.	A Flatten layer that will convert the images into an array that our model can understand.
2.	2 hidden layers that will process this array,
3.	An output layer that will return a guess at the correct class for this item.

