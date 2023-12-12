# -*- coding: utf-8 -*-
"""
Created on Thu May 18 12:48:02 2023

@author: Roshan
"""

import streamlit as st
from transformers import GPT2LMHeadModel , GPT2Tokenizer
from transformers import pipeline

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2" , pad_token_id =tokenizer.eos_token_id)


st.title("AI")

with st.sidebar :
    st.title("options")
    choice = st.radio("section", ["create blog","summarize blog"])

if choice == "create blog":
    text = st.text_input("what you want to create a blog about :")
    length = st.number_input('Enter blog size',min_value = 20,step = 1)
    button = st.button('Create')
    
    if button : 
        #st.write("perfect")
        
    
        inp_id = tokenizer.encode(text, return_tensors='pt')
        output = model.generate(inp_id , max_length = length,num_beams= 2 , no_repeat_ngram_size =2,early_stopping=True)
        op = st.write(tokenizer.decode(output[0],skip_special_tokens=True))
        #button_download = st.button("Download")
        #with open("blog.txt","w") as f :
         #   st.download_button("Done!",f , "op.txt")


if choice == "summarize blog":
    
    text = st.text_area("Enter the blog")
    sumarizer  = pipeline('summarization')
    
    sum_button = st.button("summarize")
    if sum_button : 
        article = text
        article = article.replace(".",".<eos>")
        article = article.replace("?","?<eos>")
        article = article.replace("!","!<eos>")
        sentence = article.split('<eos>')
        
        max_chunk = 500
        current_chunk = 0
        chunk = []

        for sent in sentence:
            if len(chunk) == current_chunk +1:
                if len(chunk[current_chunk]) + len(sent.split(' ')) <= max_chunk:
                    chunk[current_chunk].extend(sent.split(' '))
                else :
                    current_chunk += 1
                    chunk.append(sent.split(' '))
            else:
                chunk.append(sent.split(' '))
                
        for chunk_id in range(len(chunk)):
            
            chunk[chunk_id] = ' '.join(chunk[chunk_id])
            
        res = sumarizer(chunk , max_length =100 , min_length = 20 ,do_sample=False)
        summary = ' '.join(summ['summary_text'] for summ in res)
        
        st.write(summary)
    #st.write("sucess1")
    