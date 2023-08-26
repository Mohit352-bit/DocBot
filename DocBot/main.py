from flask import Flask, render_template, request
import openai 
import textract
import openai
from langchain.embeddings import OpenAIEmbeddings
import pinecone
from langchain.vectorstores import Pinecone
from pinecone import Index
import os 
import requests
from dotenv import load_dotenv
load_dotenv()
pinecone.init(api_key=os.getenv('PINECONE_API_KEY') , environment='us-west4-gcp-free')
openai.api_key = os.getenv('OPEN_AI_KEY')
import re



app = Flask(__name__)


## TO EXTRACT TEXT FROM FILE
# #extract text from the files
# textra = textract.process("data/demo.pdf").decode('utf-8').strip()



##funtion to chunk the file 
# def chunk_token_splitter(text):
#     # the Token text Splitter
#     from langchain.text_splitter import TokenTextSplitter
#     text_splitter = TokenTextSplitter(chunk_size=50, 
#                                     chunk_overlap=0)
#     chunks = text_splitter.split_text(text)
#     return chunks, len(chunks)


##funtion to chunk the data
# if _name_ == '_main_'

#     #pass in text from pdf to get text
#     all_pdf_content = textra

#     #Split into tokens
#     chunked_pdf_content = chunk_token_splitter(all_pdf_content)[0]

    
#function to embed
def text_embedding(text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']



## TO ADD DATA TO PINECONE
# index_name = "text-index"
# pinecone.create_index(index_name, dimension=384)

# def chunks_to_index(chunked_data):

#     #creating embedding to store embedded text and metadata to store text for reference
#     chunk_list = chunked_data
#     # embeddings =[]
#     # metadata = []
#     cnt= 1

#     for chunk in chunk_list[0:20]:

#         embedded_text = text_embedding(chunk) # embedding the text inside each chunk
#         metadata = {cnt: chunk}
    
        
#         index.upsert( [ (f"{cnt}", embedded_text , metadata)])
#         cnt =cnt+ 1

        
# chunks_to_index(chunked_pdf_content)




#funtion to retrive query
index = pinecone.Index("fin1")
gpt_meta = ''
def query_user_data(query):

    query_vec = text_embedding(query)

    #pinecone retrival query
    out = index.query(              
    vector=query_vec,
    top_k=5,
    include_values=True,
    include_metadata=True,
    score= 0.8,
    )

    #filtering the similirity search by increasing the cosine threshold
    filtered_results = [result for result in out['matches'] if result['score'] >= 0.73]

    #to extract the metadata stored in pinecone
    gpt_meta = ''
    for match in filtered_results:
        for i ,metadata in match["metadata"].items():
            gpt_meta = f"{gpt_meta} {metadata}"

    

    return gpt_meta

# storing history list
store =[]





#funtion to get the response for user query 
def send_message(message , conversations ):
    


    # conversations.append({'role':'user', 'content':f"{message}"})
    # store.append({'role':'user', 'content':f"{message}"})
    

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages= conversations,
        temperature = 0.0
        
    )
    if len(store) == 16:
        conversations.pop(1)
        conversations.pop(2)
        
    reply = response.choices[0].message.content
    # conversations.append({'role':'assistant', 'content':f"{reply}"})
    # store.append({'role':'assistant', 'content':f"{reply}"})
    
    
    return reply

def remove_plus_sign(text):
    return text.replace('+', ' ')

# def req():
#     user_input =request.get_data(as_text=True)
#     if user_input[0] == '<':
#         user_input = re.findall(r'<p>(.*?)</p>', user_input, re.DOTALL)[0]
#         user_input = remove_plus_sign(user_input)
#     else:
#         user_input = remove_plus_sign(user_input)

@app.route('/')
def home():
    return render_template('index.html')




@app.route('/process', methods=['POST'])


def process():
    
    user_input =request.get_data(as_text=True)
    if user_input[0] == '<':
        user_input = re.findall(r'<p>(.*?)</p>', user_input, re.DOTALL)[0]
        user_input = remove_plus_sign(user_input)
    else:
        user_input = remove_plus_sign(user_input)

    print(user_input)    
    test = query_user_data(user_input)
    # Process the user input and generate a response
    conversations2 = [{'role':'system', 'content':f""" 
                        You are a HR and Finance chatbot and you must answer questions realted to HR and Finance ONLY.
                        your answer should be precise and small.

                        QUESTION asked by me : {user_input}

                        Analyse whether the QUESTION asked by is realted to HR and Finance topic or not 
                        IF it is out of HR and Finance topics reply that you dont have that expertise to answer in a polite tone.
                        
                        
                            """}]

    conversations = [{'role':'system', 'content':f"""

    You are a HR and Finance Question Answering Bot. 
    You are trained on HR and Finance related data only
    You task is to give analyse the question and generate answer from the context provided.
    You can tell me about HR and Finance related quries only.
    

    Question asked by me -
    Question= {user_input}

    Provide thorough answer for the question asked by me, from the CONTEXT only-
    CONTEXT = {test}
   

    your response should always be polite, neat and friendly and should ask for futher assistance.

    """} ]
    
    # print(store)

    store.append({'role':'user', 'content':f"{user_input}"})

    
    conversations.extend(store)
    conversations2.extend(store)

    if test=='':
        conversations = conversations2


    print(conversations)




    response = send_message(user_input, conversations )
    store.append({'role':'assistant', 'content':f"{response}"})
    # print(store)
    return render_template('index.html', user_input=user_input[11:], response=response)

if __name__ == '__main__':
    app.run(debug=True ,port=8090)