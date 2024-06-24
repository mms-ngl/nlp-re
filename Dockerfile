FROM python:3.9-slim

WORKDIR /home/app

# install requirements

COPY requirements.txt .
RUN pip install -r requirements.txt

# copy model

COPY model model

# copy code

COPY re re
ENV PYTHONPATH re

# standard cmd

CMD [ "python", "re/app.py" ]
