from csv import reader
from sklearn.cluster import KMeans
import joblib
import ray


ray.init()


# Load a CSV file
def load_csv(filename):
    file = open(filename, "rt")
    lines = reader(file)
    dataset = list(lines)
    return dataset

# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())

# Convert string column to integer
def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup

def getRawIrisData():
    # Load iris dataset
    filename = 'iris.csv'
    dataset = load_csv(filename)
    print('Loaded data file {0} with {1} rows and {2} columns'.format(filename, len(dataset), len(dataset[0])))
    print(dataset[0])
    # convert string columns to float
    for i in range(4):
        str_column_to_float(dataset, i)
    # convert class column to int
    lookup = str_column_to_int(dataset, 4)
    print(dataset[0])
    print(lookup)

    return dataset

@ray.remote
def getTrainData():
    dataset = getRawIrisData()
    trainData = [ [one[0], one[1], one[2], one[3]] for one in dataset ]

    return trainData

@ray.remote
def getNumClusters():
    return 3

@ray.remote
def train(numClusters, trainData):
    print("numClusters=%d" % numClusters)

    model = KMeans(n_clusters=numClusters)

    model.fit(trainData)

    # save model for prediction
    joblib.dump(model, 'model.kmeans')

    return trainData

@ray.remote
def predict(irisData):
    # test saved prediction
    model = joblib.load('model.kmeans')

    # cluster result
    labels = model.predict(irisData)

    print("cluster result")
    print(labels)


def machine_learning_workflow_pipeline():
    trainData = getTrainData.remote()
    numClusters = getNumClusters.remote()
    trainData = train.remote(numClusters, trainData)
    result = predict.remote(trainData)

    result = ray.get(result)
    print("result=", result)



if __name__ == "__main__":
    machine_learning_workflow_pipeline()






from fastapi import FastAPI, Request

app = FastAPI()


@app.get("/items/{item_id}")
def read_root(item_id: str, request: Request):
    client_host = request.client.host
    return {"client_host": client_host, "item_id": item_id}


"""
curl --location --request POST 'http://127.0.0.1:9527/translation' --header 'Content-Type: application/json' -d '{
    "payload": {
        "fromLang": "en",
        "records": [
            {
                "id": "123",
                "text": "Life is like a box of chocolates."
            }
        ],
        "toLang": "ja"
    }
}'

8265;9527

docker stop $(docker ps -aq); docker rm $(docker ps -aq)
docker build . -t ensalty/ml-assignment -f ./ml-assignment/app/Dockerfile
docker push ensalty/ml-assignment
docker run -p 127.0.0.1:8265:8265 -p 127.0.0.1:9527:9527 --cpus=4 --shm-size=2.47gb -it ensalty/ml-assignment:latest /bin/bash

/root/.local/bin/ray start --head --dashboard-host="0.0.0.0" --num-cpus 4
RAY_task_oom_retries=10 /root/.local/bin/serve run -h 0.0.0.0 -p 9527 translate_service:translator
  

eval $(minikube docker-env) 
kubectl create -f ml-assignment/k8s/deployment.yaml


"""
