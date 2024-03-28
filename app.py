import random
from flask import Flask, render_template , request 
import qa

app = Flask(__name__, template_folder='static')


@app.route("/")
def hi():
       return render_template('qa.html', answers=[])


@app.route("/qa", methods=['GET', 'POST'])
def question():
    if request.method == 'GET':
       return hi()
    else:
       question = request.form.get('question')
       answer = qa.createResponse(question)
       return render_template('qa.html', question=question, answer=answer)

if __name__ == '__main__':
   app.run(host='0.0.0.0', port=random.randint(8000, 8000))
