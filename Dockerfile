# The builder image, used to build the virtual environment
FROM python:3.10-slim-bullseye
 
RUN apt-get update && apt-get install -y git
 
ENV HOST=0.0.0.0
ENV LISTEN_PORT 8080
EXPOSE 8080
 
WORKDIR /app
 
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
 
CMD ["streamlit", "run", "app_new_v2.py", "--server.port", "8080"]
