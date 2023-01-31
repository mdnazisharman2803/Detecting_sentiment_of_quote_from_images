# Image Sentiment Analysis based on Text
- Determining the image sentiment is a tedious task for classification algorithms, owing to complexities in the raw images as well as the intangible nature of human sentiments. Classifying image sentiments is an evergreen research area, especially in social data analytics. In current times, it is a common practice for the majority of people to precise their feelings on the web by substituting text with the upload of images via a multiplicity of social media sites like Facebook, Instagram, Twitter as well as any other platform. To identify the emotions from visual cues, some visual features, as well as image processing techniques, are used. Several existing systems have already introduced emotion detection using machine learning techniques, but the traditional feature extraction strategies do not achieve the required accuracy on random objects. In the entire process, normalization of images, feature extraction, and feature selection are important tasks in the training module. This work articulates the newest developments in the field of image sentiment employing deep learning techniques.

- Image Sentiment Analysis aims to understand how images affect people, in terms of emotions. The use of images to express views, opinions, feelings, emotions, and sentiments has increased tremendously on social platforms like Facebook, Instagram, Twitter, etc. 

- The rapid diffusion of social media platforms and their impact on our lives is increasing with each passing day. In this project, we want to find the sentiments of those texts embedded in the images. For this, we propose various model-based approaches and compare them to get the best accuracy.

 ## Introduction

- Sentiment analysis is the use of natural language processing (NLP), machine learning, and other data analysis techniques to analyze and derive objective quantitative results from the raw text. It is the process of detecting positive or negative sentiment in text. It’s often used by businesses to detect sentiment in social data, gauge brand reputation, and understand customers. Since customers are expressing their thoughts and feelings more openly than ever before, sentiment analysis is becoming an essential tool to monitor and understand that sentiment. 
Automatically analyzing customer feedback, such as opinions in survey responses and social media conversations, allows brands to learn what makes customers happy or frustrated. And accordingly, they can tailor their products and services to meet the customers’ needs.
For example, using sentiment analysis to automatically analyze 4,000+ reviews about your product could help you discover whether the customers are happy about your pricing plans and customer services or not.
Maybe you want to gauge brand sentiment on social media, in real-time and over time, so you can detect disgruntled customers immediately and respond accordingly as soon as possible.

## Advantages
-	It can be used to hide insensitive content and images from public sites.
-	It can be used to create a good feeling and habits in a child by showing them the positive contents. 
-	Negative images create a negative feeling in the mind and a person gets attracted towards it. By applying Image sentiment analysis, we can reduce the amount of crime.
-	It can be used to read the minds of people and help in consulting them.

## Disadvantages
-	This can be used to incite the feelings of mass in wrongdoings.
-	It may lead to the exploitation of the customers.

## Methodology

- In this research first, we have to get the text from the image. Using OCR, we can get the text on the image. OCR can be implemented using pytesseract, and we will use deep learning models as well. After getting the text of the images, half of the work is done and now, we will implement sentiment processing.

### OCR

   - Optical character recognition or optical character reader (OCR) is the electronic or mechanical conversion of images of typed, handwritten, or printed text into machine-encoded text, whether from a scanned document, a photo of a document. In today’s world rather than storing data on papers and books, we are storing them digitally as computer hardware now can store terabytes of data so we had no problem storing millions of books in a 1 Gigabyte of the harddisk. OCR helps us to scan the images, documents, or any paper document and convert their text into the digital text form, rather than a handwritten or computer-generated receipt. OCR as a process generally consists of several sub-processes to perform as accurately as possible. In programming languages for OCR, we had the tesseract API. It was considered one of the most accurate open-source OCR engines then available. It was originally developed by Hewlett-Packard and later on Google sponsored their development and improved it a lot. In this article, we will work through how we use Pytesseract to read the various scanned documents using Python.

###	Pytesseract

    - Python-tesseract is an optical character recognition (OCR) tool for python. That is, it will recognize and “read” the text embedded in images.
Python-tesseract is a wrapper for Google’s Tesseract-OCR Engine. It is also useful as a stand-alone invocation script to tesseract, as it can read all image types supported by the Pillow and Leptonica imaging libraries, including jpeg, png, gif, bmp, tiff, and others. Additionally, if used as a script, Python-tesseract will print the recognized text instead of writing it to a file.

###	Keras

- keras-ocr provides out-of-the-box OCR models and an end-to-end training pipeline to build new OCR models. Keras-OCR is an image-specific OCR tool. If text is inside the image and their fonts and colors are unorganized, Keras-ocr gives good results.

 

## Text Preprocessing

- Text preprocessing is a method to clean the text data and make it ready to feed data to the model. Text data contains noise in various forms like emotions, punctuation, text in a different case. When we talk about Human Language then, there are different ways to say the same thing, and this is only the main problem we have to deal with because machines will not understand words, they need numbers so we need to convert text to numbers in an efficient manner.

###	Count Vectorization

- Count Vectorization involves counting the number of occurrences each word appears in a document (i.e distinct text such as an article, book, even a paragraph!). Python’s Sci-kit learn library has a tool called CountVectorizer to accomplish this.
Example sentence: “The weather was wonderful today and I went outside to enjoy the beautiful and sunny weather.” You can tell from the output below that the words “the”, “weather”, “and “and” appeared twice while other words appeared once. That is what Count Vectorization accomplishes.

-	TF-IDF

TF-IDF stands for “Term Frequency — Inverse Document Frequency”. This is a technique to quantify a word in documents, we generally compute a weight to each word which signifies the importance of the word in the document and corpus. This method is a widely used technique in Information Retrieval and Text Mining.
If I give you a sentence for example “This building is so tall”. It’s easy for us to understand the sentence as we know the semantics of the words and the sentence. But how will the computer understand this sentence? The computer can understand any data only in the form of numerical value. So, for this reason, we vectorize all of the text so that the computer can understand the text better.

-	Bag of Words

A bag-of-words model, or BoW for short, is a way of extracting features from the text for use in modeling, such as with machine learning algorithms.
The approach is very simple and flexible and can be used in a myriad of ways for extracting features from documents.
A bag-of-words is a representation of text that describes the occurrence of words within a document. It involves two things:



## Model Training:

###	MultinomialNaiveBayes

   - Multinomial Naive Bayes algorithm is a probabilistic learning method that is mostly used in Natural Language Processing (NLP). The algorithm is based on the Bayes theorem and predicts the tag of a text such as a piece of email or newspaper article. It calculates the probability of each tag for a given sample and then gives the tag with the highest probability as output.
Naive Bayes classifier is a collection of many algorithms where all the algorithms share one common principle, and that is each feature being classified is not related to any other feature. The presence or absence of a feature does not affect the presence or absence of the other feature.

###	Long short term memory

   - Long Short Term Memory is a kind of recurrent neural network. In RNN output from the last step is fed as input in the current step. LSTM was designed by Hochreiter & Schmidhuber. It tackled the problem of long-term dependencies of RNN in which the RNN cannot predict the word stored in the long-term memory but can give more accurate predictions from the recent information. As the gap length increases RNN does not give an efficient performance. LSTM can by default retain the information for a long period of time. It is used for processing, predicting, and classifying on the basis of time-series data. 

###	Hyperparameter Tuning

   - A hyperparameter is a parameter whose value is set before the learning process begins. When creating a machine learning model, you'll be presented with design choices as to how to define your model architecture. Oftentimes, we don't immediately know what the optimal model architecture should be for a given model, and thus we'd like to be able to explore a range of possibilities. In true machine learning fashion, we'll ideally ask the machine to perform this exploration and select the optimal model architecture automatically. Parameters that define the model architecture are referred to as hyperparameters and thus this process of searching for the ideal model architecture is referred to as hyperparameter tuning.

## Conclusion 
Sentiment analysis is an emerging technique and the social network content can help understand user behavior and provide useful information for related data analysis. Sentiment analysis is very important for data analysis of social behavior. We discussed various methods that we used to perform image sentiment analysis and to obtain the best accuracy and we found a suitable approach.

## Minor Project ECE - 2023 :

<h2 align="center">Submitted by </h2>

<table align="center">
	<tr>
		<td>
		<a href="https://github.com/mdnazisharman2803"><img  src="https://user-images.githubusercontent.com/98539013/192952215-c2bf7950-93eb-4cf5-80ae-0b3dc95b3754.png" width=200px height=150px /></a></br> <h4  style="color:blue">Md Nazish Arman</h4><br>
		</td>
    <td>
		<a href=""><img src="" width=150px height=150px /></a></br> <h4 style="color:limegreen;"></h4><br/>
		</td>
	</tr>
</table>
<br>

## Working Video
- https://user-images.githubusercontent.com/98539013/215455632-a00deb77-5ddf-46fd-8017-d78d4d70290a.mp4

## References

-	https://www.cc.gatech.edu/~hays/7476/projects/Aditi_Vasavi.pdf
-	https://www.scitepress.org/Papers/2019/79096/79096.pdf
-	https://ieeexplore.ieee.org/document/8609672
-	https://ijesc.org/upload/551d641a6711d6c0c0deadacb3cfc168.Image%20Sentiment%20Analysis.pdf
-	https://medium.com/mlearning-ai/tesseract-vs-keras-ocr-vs-easyocr-ec8500b9455b
-	https://levelup.gitconnected.com/a-beginners-guide-to-tesseract-ocr-using-pytesseract-23036f5b2211
-	https://towardsdatascience.com/sentiment-analysis-concept-analysis-and-applications-6c94d6f58c17




