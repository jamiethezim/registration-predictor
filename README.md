# Account Registration Model Training

This is a tool that predicts if someone creating an account is likely to be a client or rather a confused end-user. The markers are unique enough that this model would never be productized, as it's a vastly over-engineered solution to what could not be more of a simple problem. We can almost immediately (and probably programmatically) tell from inferring the email address if the person is a client or consumer.

Nevertheless, it's good to get reps in.

Here I've trained a model on features that are available on account signup and can predict (with low accuracy) if a person is someone who might pay money for our services in the future (client) or is just someone looking to get verified (consumer).

I revisited ML concepts during my company's HackWeek in November of 2022. I collected a sample of 169 records, 25 client and 144 consumers. For internal reasons I had a very difficult time pulling records and collecting a balanced dataset. I classified each by using my eyeballs, but don't want to expose what criteria I used to categorize (though it is largely related to their email address).

Here is the output from my code:

```
Shape: (169, 7)

Features: Index(['userId', 'accountId', 'username', 'company', 'companyUrl', 'email',
'identifier'],
dtype='object')

Feature matrix:
	userId 			 accountId 		  username 	company      companyUrl     email
0       638431b9054f065ad78c0691 638431b9054f065ad78c0691 {obfuscated}  {obfuscated} {obfuscated}   {obfuscated} 
...

Response vector:
['client' 'client' 'client' 'client' 'client' 'client' 'client']

Training model...
--------------------------------------------------

Model Accuracy score is:  0.7941176470588235

Feature Importance:  	0
companyUrl  		0.233855
accountId 		0.177590
userId  		0.177030
company 		0.151584
email 			0.147649
username  		0.112291
  

Classification Report:
		 precision  recall  f1-score  support
client 	 	 0.00  	    0.00    0.00      6
consumer 	 0.82       0.96    0.89      28

accuracy 			    0.79      34
macro avg    	 0.41       0.48    0.44      34
weighted avg 	 0.67       0.79    0.73      34

Confusion Matrix:
		client   consumer
client 		0 	 6
consumer 	1  	 27
```
## Teck Stach
I used python libraries to write this model trainer. I used pandas to parse the data and scikit and numpy libraries to split the data into train and test sets, build a random forest classifier model, train it, and report on its accuracy.

## Description of Files

- `model_builder.py`: execute this python program to train a model and report results on the model's accuracy
- `example_input.csv`: an example input file that can be passed in and the model trained on. The file I used contains PII, this file is just to show formatting.
- `requirements.txt`: virtual environment set up
