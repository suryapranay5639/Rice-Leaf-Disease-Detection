import os
import sqlite3
import numpy as np
import tensorflow as tf
import gc
from io import BytesIO  # <--- THIS IS THE MISSING LINE
from flask import Flask, render_template, request, make_response
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
from xhtml2pdf import pisa
from disease_info import disease_details

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

MODEL_PATH = 'model_final.h5'
model = None

# Final 8 Rice Classes
CLASS_NAMES = [
    "Bacterial Leaf Blight", "Brown Spot", "Healthy Rice Leaf", 
    "Leaf Blast", "Leaf scald", "Narrow Brown Leaf Spot", 
    "Rice Hispa", "Sheath Blight"
]

def load_model_safely():
    global model
    if model is None:
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"--- Model Loaded: {model.input_shape} ---")
    return model

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global model
    if model is None: load_model_safely()
    
    file = request.files.get('file')
    if not file or file.filename == '': return "No file", 400

    filename = file.filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        # EfficientNetB3 Target Size: 300x300
        img = image.load_img(filepath, target_size=(300, 300))
        img_array = image.img_to_array(img)
        
        # TTA: Flip and Rotate to verify accuracy
        tta_batch = np.array([img_array, np.fliplr(img_array), np.flipud(img_array), np.rot90(img_array)])
        tta_batch = preprocess_input(tta_batch)

        preds = model.predict(tta_batch, verbose=0)
        avg_preds = np.mean(preds, axis=0)
        idx = np.argmax(avg_preds)
        
        prediction = CLASS_NAMES[idx]
        confidence = round(float(avg_preds[idx]) * 100, 2)

        return render_template("result.html", 
                               prediction=prediction, 
                               confidence=confidence,
                               image_path=f"uploads/{filename}", 
                               info=disease_details.get(prediction, {}))
    except Exception as e:
        return f"Error: {str(e)}", 500
    finally:
        gc.collect()

@app.route('/download_report')
def download_report():
    prediction = request.args.get('prediction')
    confidence = request.args.get('confidence')
    info = disease_details.get(prediction, {})

    html = f"""
    <html>
    <head>
        <style>
            body {{ 
                font-family: 'Helvetica', 'Arial', sans-serif; 
                padding: 40px; 
                color: #333;
                line-height: 1.6;
            }}
            .header {{ 
                text-align: center; 
                border-bottom: 3px solid #27ae60; 
                padding-bottom: 10px;
                margin-bottom: 30px;
            }}
            .header h1 {{ color: #27ae60; margin: 0; font-size: 24pt; }}
            .header p {{ font-style: italic; color: #7f8c8d; }}
            
            .result-box {{ 
                background-color: #f9f9f9; 
                padding: 20px; 
                border-radius: 10px;
                border: 1px solid #ddd;
                margin-bottom: 20px;
            }}
            .label {{ font-weight: bold; color: #2c3e50; width: 150px; display: inline-block; }}
            
            h3 {{ 
                color: #27ae60; 
                border-bottom: 1px solid #27ae60; 
                padding-bottom: 5px;
                margin-top: 30px;
            }}
            .recommendation {{ margin-left: 10px; padding: 10px; background: #fff; }}
            
            .footer {{ 
                margin-top: 50px; 
                font-size: 9pt; 
                text-align: center; 
                color: #95a5a6;
                border-top: 1px solid #eee;
                padding-top: 20px;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Rice Health Diagnosis Report</h1>
            <p>Automated Plant Pathology Analysis System</p>
        </div>
        
        <div class="result-box">
            <p><span class="label">Diagnosis:</span> {prediction}</p>
            <p><span class="label">Confidence:</span> {confidence}%</p>
            <p><span class="label">Status:</span> {"Attention Required" if "Healthy" not in prediction else "Normal"}</p>
        </div>

        <h3>Treatment Recommendations</h3>
        <div class="recommendation">
            <p><strong>Organic Approach:</strong><br>{info.get('org_en', 'N/A')}</p>
            <p><strong>Chemical Intervention:</strong><br>{info.get('chem_en', 'N/A')}</p>
        </div>
        
        <div class="footer">
            <p>This report was generated using a Deep Learning model (EfficientNetB3) trained on rice leaf imagery.<br>
            <strong>Disclaimer:</strong> This is for research and preliminary diagnostic purposes only.</p>
        </div>
    </body>
    </html>
    """
    
    pdf_out = BytesIO()
    # Simplified encoding for standard English PDF
# Change "Arial" to "utf-8"
    pisa_status = pisa.CreatePDF(BytesIO(html.encode("utf-8")), dest=pdf_out)    
    if pisa_status.err:
        return "Error generating PDF", 500
        
    response = make_response(pdf_out.getvalue())
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = f'attachment; filename=Diagnosis_Report.pdf'
    return response

@app.route('/history')
def history():
    conn = sqlite3.connect('history.db')
    c = conn.cursor()
    # Ensure the order matches scan[0], scan[1], etc. in your HTML
    c.execute("SELECT image_path, label, confidence, date FROM predictions ORDER BY date DESC")
    data = c.fetchall()
    conn.close()
    return render_template('history.html', scans=data)

@app.route('/encyclopedia')
def encyclopedia():
    # Uses the data already in disease_info.py
    return render_template('encyclopedia.html', diseases=disease_details)

@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)