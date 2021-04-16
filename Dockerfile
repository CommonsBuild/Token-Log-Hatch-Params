FROM python:3.6

WORKDIR .

COPY . .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -e ./
