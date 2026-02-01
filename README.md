# Project: From Model to Production

A System that can detect and classify images belongs to fashion products. The Classification model is deployed as FastAPI service that runs overnight, and process the images in batches.

<p align="center" width="40">
  <img src="https://github.com/jamesantonydas/Refund_Department_Classification/blob/main/images/mod2prod_title.png"/>
</p>

## Getting Started

This project needs the python3 or above runtime installed in your system. 
You can download and install the latest version of python from python's official website.

### Downloading the application

Please download the application files from this repository, or create a new folder and simply run the command,

```
git clone https://github.com/jamesantonydas/Refund_Department_Classification
```

### Running the Docker API Service

After downloading the application, please run the following command inside the newly created directory to build the docker image

```
docker build -t fashion-api .
```

To run the docker service, please run,


```
docker run -p 8000:8000 fashion-api
```

This will start the service at localhost:8000

### Testing the Application

You can manually test this application by running

```
python sender.py
```

### Setting up the CronJob

The system can be scheduled to run everymidnight midnight

```
0 0 * * * /usr/bin/python3 /home/user/path/to/sender.py
```

Or you can setup Github action using the schedule.yaml file
