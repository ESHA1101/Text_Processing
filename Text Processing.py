
from cleantext import clean
f = open("D:\ml.txt",'r', encoding = 'utf-8')
text = f.read()

from nltk import tokenize
s=tokenize.sent_tokenize(text)

emoji=clean(s, no_emoji=True)

#define special characters list
special_characters = ['!','#','$','%', '&','@','[',']',']','_','^','(',')','-','\\','っ','❛' ,'ᴗ','❛','...']

#using for loop and replace to remove special characters
for i in special_characters:
    emoji = emoji.replace(i,'')
    
final_emoji=list(emoji.split(","))

import pandas as pd
df=pd.DataFrame(final_emoji)
df.columns=['Sentences']

from textblob import TextBlob
senti=[]
for i in final_emoji:
    testimonial = TextBlob(i)
    senti.append(testimonial.sentiment.polarity)
df['Polarity']=senti

from textblob import TextBlob
subj=[]
for i in final_emoji:
    testimonial = TextBlob(i)
    subj.append(testimonial.sentiment.subjectivity)
df['Subjectivity']=subj

word=[]
for i in final_emoji:
    sent=TextBlob(i)
    word.append(len(sent.words))

df['Sentence_length']=word
print(df)




