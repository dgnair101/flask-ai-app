from flask import Flask, render_template, request, jsonify
import ollama

app=Flask(__name__)

conversation_history=[]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat",methods=["POST"])
def chat():
    user_message=request.json["message"]

    #Add user msg to history
    conversation_history.append({
        "role":"user",
        "content":user_message
    })

    #Send full history to ollama

    response=ollama.chat(model="llama3.2", messages=[{"role":"system", "content":"You are a helpful AI assistant. Be concise and clear"},
        ] + conversation_history
    )

    ai_response = response["message"]["content"]

    #Add AI response to history
    conversation_history.append({
         "role":"assistant",
        "content":ai_response
    })
    return jsonify({"response":ai_response})
def clear():
       conversation_history.clear()
       return jsonify({"status":"cleared"}) 

if __name__== "__main__":
    app.run(debug=True)
