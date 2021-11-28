# 1. Library imports

from numpy import double
from pydantic import BaseModel 
import uvicorn
from fastapi import FastAPI
from yoloclass import Yolo




# 2. Create the app object
app = FastAPI()

model= Yolo()

class Data(BaseModel):
    path: str #image path

# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    '''
    This is a first docstring.
    '''
    return {'message': 'Hello, strange'}

# 4. Route with a single parameter, returns the parameter within a message
#    Located at: http://127.0.0.1:8000/AnyNameHere
@app.post('/yolov4/')
def yolov4(params: Data):
    '''
    Corre una arquitectura Yolo V4

    '''
    print('en yolo')

    #img_path='data\images\dog.jpg'
    #boxes, scores, classes, num_objects, no_object = model.yolo_v4(img_path)

    
    boxes, scores, classes, num_objects, no_object = model.yolo_v4(params.path)


    print('lo paso')
    #print (x_min,y_min,x_max,y_max) 

    return {
                'boxes': boxes,
                'score': scores,
                'classes': classes,
                'num_objects': num_objects,
                'no_object': no_object,
          }
  


# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8001)   