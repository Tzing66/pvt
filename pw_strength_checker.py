#test script for initial ML learning. Too embarrassed to add the date.

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

#change the directory before using the code
data = pd.read_csv("C:/Users/tanma/Downloads/data_pass/data.csv", on_bad_lines='skip')
print(data.head())

data = data.dropna()

#mapping values
data["strength"] = data["strength"].map({0: "Weak", 1: "Okay",2: "Strong"})

print(data.sample(5))
print(data.head())

#defining a function to breakdown the passwords
def word(pw):
    chars = []
    for i in pw:
        chars.append(i)
    return chars


#converting password and strength to individual arrays (for later)
x = np.array(data["password"])
y = np.array(data["strength"])


#Initializing TfidfVectorizer object. The TfidfVectorizer is a tool used to convert a collection of text documents into a matrix (tf-idf matrix which basically quantifies the inportance/releavnace of strings). 
# Tokenizer = word is us specifying what function to use instead of its own tokenizer.
tdif = TfidfVectorizer(tokenizer=word) 
print("flag1")

#Applies the vectorizer onto passwords array, transforming password strings into tf-idf matrix (where each row corresponds to a password, and each column corresponds to a TF-IDF score for a character)
x = tdif.fit_transform(x) 
print("flag2")
#x becomes the data, y are the "labels"(password strength). test size sets test data as 5% of the total, rest is training. 
# random state = 42 makes it reproducible. This gives us the trianing and testing sets
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.05, random_state=42)
print("flag3")

#RFC uses decision treets to make predictions. Each DT gives a prediction and (usually) a majority is taken.

model = RandomForestClassifier() #initial model, super slow. acc = 95%. But like SUPER slow.

model2 = RandomForestClassifier(n_jobs=-1, max_features='sqrt', n_estimators=100, max_depth=10) #stuff i found whcih might increase the speed. acc = 83% (time was DRAMATICALLY lesser)
model2.fit(xtrain, ytrain)
print(model2.score(xtest, ytest))


##notes - this code slow af. idk how to speed it up. i assumed this is such a small code so it should be fine but it clearly isnt. 
# Model 2 was slightly better but there was a huge loss in accuracy.

##You can use the below section to check a custom password and its strength! Change model as needed.
import getpass
user = getpass.getpass("Enter Password: ")
data = tdif.transform([user]).toarray()
output = model2.predict(data)
print(output)
