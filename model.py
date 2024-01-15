from keras.layers import Activation, Dense
from keras.preprocessing.text import  Tokenizer
from keras.models import   Sequential, load_model
import numpy as np
# input data
bot_replies=[]
seller_queries=[]
with open("Conversation.txt") as f:
    data=f.readlines()
#separate data
max_sentence_len=len(max(data,key=len))# store maximum length of string
for i,d in enumerate(data):#sseparate bots answer and seller queries
     if  i%2 !=0: 
             bot_replies.append(d)
     else:
             seller_queries.append(d)   
vocablen=len(bot_replies)

#2.1 Convert separated lines into embedded matrix

t=Tokenizer(num_words=max_sentence_len, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split='\n', char_level=False, oov_token=True)
t.fit_on_texts(bot_replies)
word_index=t.word_index
index_to_word=t.index_word
def Matrix_Maker(unprocessed_lines):
        t.fit_on_texts(unprocessed_lines)
        matrix=t.texts_to_matrix(unprocessed_lines,mode='binary')
        return matrix 

def Seq2Seq():
    #Define Sequential Model
    model = Sequential() 
    #Create input layer
    model.add(Dense(max_sentence_len, input_shape=(max_sentence_len,)))
    model.add(Dense(max_sentence_len))
    model.add(Dense(max_sentence_len))
    #Create hidden layer 
    model.add(Activation('relu')) 
    #Create Output layer
    model.add(Activation('sigmoid')) 
    #model.compile(optimizer='rmsprop',loss='mse')
    model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='rmsprop')
    return model
def Trainer(model,inp,output):
    print("inp:",inp.shape)
    print("output:",output.shape)
    model.fit(inp,output, epochs=100, batch_size=32)
    return model


def PredictReply(inp):
    try:
        model=load_model("model.hdf5")
        enc_inp=Matrix_Maker(inp)
        prediction=model.predict(enc_inp)
        response=np.argmax(prediction,axis=1)
        word= [index_to_word[a%vocablen+4] for a in response]
        sentence=word[1]
        return sentence
    except (Exception, ValueError,IndexError) as e:
        return ["Error[001]!cannot procced. Please try again later",e]    
inp=Matrix_Maker(bot_replies)
output=Matrix_Maker(seller_queries)

def UpdateModel():
        model=Seq2Seq()
        model=Trainer(model,inp,output)        
        model.save("model.hdf5")
        return Null

try:
    model=load_model   
except:
    model=Seq2Seq()
    model=Trainer(model,inp,output)
    model.save("model.hdf5")
    
#Hey
# Model_maker()   

    
