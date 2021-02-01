# 1. Library imports
import uvicorn
import pickle
import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel


class Rent(BaseModel):
    rooms: int
    location: str 


# 2. Create app and model objects
app = FastAPI()
model = joblib.load("joblib_model.sav")

# 3. Expose the prediction functionality, make a prediction from the passed

@app.get("/")
def read_root():
    return {"Hello": "Stutern"}

@app.post("/predict/")
def predict_rent(data: Rent):
    data = data.dict()
    rooms = data['rooms']
    location = data['location']
    result = model.predict([rooms, location])
    result = np.exp(result)
    
    return {'Expected rent is': result}

    '''
    if result:
        return {"statusCode": 200,
                "body": {"Expected rent": result}}
    else:
        return {"status": 404,
                "body": {"Message": "Are you sure you're using the right data ?"}}
                '''
                

    
# 4. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
