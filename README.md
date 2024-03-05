# Fake-News-Detection
Objective:  To create a machine learning model in order to classify the given news as fake or real using RNN(Recurrent Neural Network).
-------------------------------------------------------------------------------------------------------
Approach to solving the problem: 
--------------------------------
Two datasets are taken here viz. True and fake. The model is then trained on the normalized data .RNN model is created and batch processing is done in order to achieve higher efficiency and accuracy of the model keeping the consideration in mind that the GPUs work efficiently on data processed in batches.
Finally, the model evaluation is done on the testing set. Accuracy,Precision and Recall is calculated on testing set.
Manual testing is done then in order to check if the model is able to predict the label well.
The data was then split into training and testing.

-----------------------------------------------------------------------------------------------------------------
Model summary:  
-----------------
1. Import the datasets using Pandas read_csv method.
Identify them as :True_csv and Fake_csv
2.Import the necessary libraries.
3.Perform Data Pre-processing: 
a) Check for any null values in the datasets.
b) Since we wish both the dataframes to have similar distribution,check for unique values in subject column ,drop the ones that you feel could influence the accuracy of the model.
4.So,we drop Date and Subject column here.
5. Define the target variable as Class here. For fake the value would be 0 and for Real it will be 1.
6.Plot a bar graph in order to check the distribution of real and fake news.
7. We merge both the datasets together and form a new dataframe.
8.Then combine the title with text for entire dataset.
9. Then we split the dataset into training and testing.
10. Then we normalise both the training and testing datasets(remove extra spaces, lower case and url links in the data.)
11. Tokenize the text into vector.
12.Apply padding so we have articles of same length.There’s a point to note here,
RNNs can work efficiently with texts of arbitrary length,still here we applied padding to achieve a similar length for all articles. This is done in order to provide for batch processing. The training dataset is divided into batches ,for eg. here the batch size is 30 ,which means the dataset is divided into the batches of sample size 30.
In this case, a batch size of 30 means that the model will process 30 samples at a time during each training iteration.During training, the training dataset (X_train and y_train) is divided into batches, with each batch containing the specified number of samples (batch_size). The model computes the gradients and updates the weights based on the loss calculated on each batch. This process repeats for the entire training dataset over multiple epochs.
13.Build the RNN model.
14. We use early stopping, which stops when the validation loss no longer improves.
----------------------------------------------------------------------------------------------------------------
Results: 
  Accuracy on testing set: 0.9894209354120267
	Precision on testing set: 0.990968040759611
	Recall on testing set: 0.9870818915801615
-----------------------------------------------------------------------------------------------------------------
Inference: The inference was drawn in terms of classification of fake news and real news out of the trained model.
Limitations: I feel ,talking of limitations is very important when it comes to NLP projects. Due to scarcity of training data in terms of news related to India ,the model could not be well trained based on Indian news. Therefore, the model does not perform well on Indian national news which is out of context.
------------------------------------------------------------------------------------------------------------------
References:  https://builtin.com/data-science/recurrent-neural-networks-and-lstm
------------------------------------------------------------------------------------------------------------------

Other comments (which may include how to further fine-tune):

In order to further fine tune the model following could be done:
1.	Create Synthetic data in order to reduce the bias.
2.	Hyperparameter tuning.
