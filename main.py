import pandas as pd
import keras as ks
import numpy as np
from keras.optimizers import SGD
import hashlib
import tensorflow as tf
start_path = "DataSource\\"
#region frames
body_frame = pd.read_csv(start_path+"body.csv", error_bad_lines=False, delimiter=';')
body_symtom_pivot_frame = pd.read_csv(start_path+"body_symptom.csv", delimiter=',')
disease_frame = pd.read_csv(start_path+"disease.csv", delimiter=',', escapechar='\\')
disease_symptom_pivot_frame = pd.read_csv(start_path+"disease_symptom.csv")
disease_sympot_body_pivot_frame = pd.read_csv(start_path+"disease_body_symptom.csv")
doc_spec_frame = pd.read_csv(start_path+"doc_spec.csv")
doctor_disease_pivot_frame = pd.read_csv(start_path+"doctor_diseases.csv")
sepciality_frame = pd.read_csv(start_path+"specialty.csv", escapechar='\\')
symptom_frame = pd.read_csv(start_path+"symptom.csv",delimiter=',')
#endregion
#print(body_frame)

symptom_frame["body_id"] = pd.to_numeric(symptom_frame["body_id"],downcast='integer')
temp_frame = body_symtom_pivot_frame.merge(body_frame, left_on='body_id',right_on='id').merge(symptom_frame,left_on='symptom_id', right_on='id')

#print(temp_frame)

disease_current_frame = temp_frame.merge(disease_symptom_pivot_frame, left_on='id', right_on='symptom_id').merge(disease_frame, left_on='disease_id', right_on='id')
print(disease_current_frame["parent"])

disease_current_frame.to_csv("new.csv")

new_frame = pd.read_csv("new.csv",delimiter=',',escapechar='\\')
print(new_frame["parent"])
#print(first_frame.clumns)
encoders = {'parent' : lambda parent : hash(parent),
            "name_x" : lambda name: hash(name),
            "deleted" : lambda deleted : hash(deleted),
            "name_y" : lambda name_y : hash(name_y),
            "deleted_y" : lambda deleted_y : hash(deleted_y),
            "gender_x" : lambda gender_x : {"male": 0, "female": 0.5, 'all' : 1}.get(gender_x),
            "name" : lambda name : hash(name),
            "alias" : lambda alias : hash(alias),
            "gender_y" : lambda gender_x : {"male": 0, "female": 0.5, 'all' : 1}.get(gender_x),
            "gender": lambda gen: {"male": 0, "female": 0.5, 'all' : 1}.get(gen),
            "about" : lambda about: hash(about),
            "description": lambda desc: hash(desc)}
input_names = ['parent', 'name_x', 'deleted','name_y','deleted_y','gender_x','name','alias','gender_y', 'deleted']
def hash(string):
    if isinstance(string, str):
        return float(int(hashlib.sha256(string.encode('utf-8')).hexdigest(), 16) % 10**8) / 100000000
    
    return float(string) / 100

def dataframe_to_dict(df):
    result = dict()
    for column in df.columns:
        values = new_frame[column].values
        result[column] = values
    return result

def make_supervised(df):
    first_frame = new_frame[input_names]
    second_frame = new_frame[['about', 'description']]
    return {"inputs": dataframe_to_dict(first_frame),
            "outputs":dataframe_to_dict(second_frame)}

def encode(data):
    vectors = []
    for data_name, data_values in data.items():
        encoded = list(map(encoders[data_name], data_values))
    
        vectors.append(encoded)
    formated = []
    for vector_raw in list(zip(*vectors)):
        vector = []
        for element in vector_raw:
            if element == "None" or element is None:
                element = float(0)
            print(element)
            vector.append(float(element))
        formated.append(vector)
    return formated

supervised = make_supervised(new_frame)

structed_inputs = encode(supervised["inputs"])
structed_outputs = encode(supervised["outputs"])
train_x = tf.convert_to_tensor(np.array(structed_inputs[: 2000], dtype=np.float))
train_y = tf.convert_to_tensor(np.array(structed_outputs[: 2000], dtype=np.float))

test_x = np.array(structed_inputs[2000:],dtype=np.float )
test_y = np.array(structed_outputs[2000:], dtype=np.float)

opt = SGD(lr=0.01, momentum=0.9)
model = ks.Sequential()
model.add(ks.layers.core.Dense(units=10, activation="relu"))
model.add(ks.layers.core.Dense(units=2, activation="sigmoid"))
model.compile(loss="mse", optimizer=opt, metrics=["acc"])

fit_results = model.fit(x=train_x, y=train_y, validation_data=(test_x, test_y), epochs = 100, validation_split = 0.2)

predicted_date = model.predict(test_x)
real_data = new_frame.iloc[2000:][input_names]

print(predicted_date)
print(real_data)