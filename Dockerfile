FROM ubuntu:latest

RUN apt update \
    && apt install -y htop python3-dev python3-pip

COPY ./requirements.txt /src

RUN pip install -r requirements.txt

COPY . /src

WORKDIR /src

EXPOSE 8080

CMD ["python3", "deploy.py"]

