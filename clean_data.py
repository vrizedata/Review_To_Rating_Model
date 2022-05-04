import re

def clean_text(text):
    # remove html tags
    # text = re.sub('<.*?>', '', text)
    # replace '-' and '.' with single spacw
    text = re.sub('[-.]',' ',text)
    #remove special charecters and numbers
    docs = [re.sub('[^A-Za-z]','',word) for word in text.split()]
    #remove extra spaces
    docs = ' '.join([doc.strip() for doc in docs])
    docs = docs.strip().lower()
    docs = re.sub('\s+',' ', docs)
    return docs