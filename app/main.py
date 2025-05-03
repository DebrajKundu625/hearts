from fastapi import FastAPI, UploadFile, File
import pandas as pd
from app.model import model,encoder
from io import StringIO
app = FastAPI()
@app.get("/")
def home():
    return {"message": "Heart disease prediction is working"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        return {"message": "Please upload csv file"}
    
    contents = await file.read()
    data_str=contents.decode("utf-8")
    new_data=pd.read_csv(StringIO(data_str))

    catagorical_cols=["Result","Risk_Level"]  
    new_data_encoded_array=encoder.transform(new_data[catagorical_cols])
    new_data_encoded=pd.DataFrame(new_data_encoded_array,columns=catagorical_cols)

    featured_cols=["Age","Gender","Heart rate","Systolic blood pressure","Diastolic blood pressure","Blood sugar","CK-MB","Troponin","Result","Risk_Level"]
    numeric_cols=[	"Age","Gender","Heart rate","Systolic blood pressure","Diastolic blood pressure","Blood sugar","CK-MB","Troponin"]
    new_data_numeric=new_data[numeric_cols]
    new_data_featured=pd.concat([new_data_numeric,new_data_encoded],axis=1)
    new_data_featured=new_data_featured[featured_cols]

    prediction=model.predict(new_data_featured)
    new_data["Recommendation"]=prediction
    return new_data[["Age","Recommendation"]].to_dict(orient="records")