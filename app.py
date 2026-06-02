from flask import Flask, render_template, request, jsonify
from groq import Groq
from dotenv import load_dotenv
from rag import load_and_index_data,retrieve_relevant_chunks
import os

load_dotenv()

app=Flask(__name__)

client=Groq(api_key=os.getenv("GROQ_API_KEY"))

index, chunks = load_and_index_data("consulting_data.xlsx")

conversation_history=[]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat",methods=["POST"])
def chat():
    user_message=request.json["message"]

    relevant_chunks = retrieve_relevant_chunks(user_message,index,chunks)

    context = "\n\n".join(relevant_chunks)


    #Add user msg to history
    conversation_history.append({
        "role":"user",
        "content":user_message
    })

    #Send full history to ollama

    response=client.chat.completions.create(model="llama-3.3-70b-versatile", 
    messages=[{"role":"system", "content":f"""You are a strategy consultant with deep expertise. So, be very structured, concise and clear. Answer questions only on the following context from our consulting knowledge base. If answer is not 
    in the context, say that you do not have the information. You have no other information. Do not make up information.
    
    Rules you must follow:
    1. only answer using information explicitly stated in the context
    2. do not infer or interpret or make up anything
    Context: {context}"""},
        ] + conversation_history, temperature=0
    )

    ai_response = response.choices[0].message.content

    #Add AI response to history
    conversation_history.append({
         "role":"assistant",
        "content":ai_response
    })
    return jsonify({"response":ai_response})

@app.route("/clear",methods=["POST"])
def clear():
       conversation_history.clear()
       return jsonify({"status":"cleared"}) 

if __name__== "__main__":
    app.run(debug=True)
