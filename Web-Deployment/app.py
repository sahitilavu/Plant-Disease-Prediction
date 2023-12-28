# import requests
# import base64
import os

from flask import Flask, render_template, request
from gradio_client import Client
from labels import classes

client = Client("https://jkompalli-trained-prediction.hf.space/--replicas/r7j7b/")
app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def home():
    return render_template('index.html')

# NOTE: Comment line 9 & 29-32 if you are uncommenting the commented lines
@app.route("/predict", methods = ['GET','POST'])
def predict():
    if request.method == 'POST':
        file = request.files['image']
        # data = base64.b64encode(file.read()).decode('utf-8')
        # file_data = "data:image/jpg;base64," + data
 
        try:
            # response = requests.post("https://jkompalli-pdd.hf.space/run/predict", json={
            #     "data": [ file_data ]
            # }).json()
            
            image_path = "temp.jpg"
            with open(image_path, "wb") as f:
                f.write(file.read())
                
            response = client.predict(image_path, api_name="/predict")
            confidence = round(response['confidences'][0]['confidence'] * 100, 2)
            response = int(response['label'].split(':')[0])
            os.remove(image_path)

        except Exception as e:
            print("Error: ", e)
            return render_template('index.html')
        
        return render_template('analysis.html', confidence = confidence,
                               prediction = classes[response]['name'],
                               image_name = f"images/{classes[response]['id']}.jpg",
                               description = classes[response]['description'],
                               treatment = classes[response]['treatment'],
                               pesticides = classes[response]['suggested_pesticides'])
    
if __name__ == "__main__":
    app.run(threaded=False,port=8080) 