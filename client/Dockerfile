FROM python:3.7-slim

WORKDIR /app
COPY app/ /app

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt \
        --no-cache-dir \
        --disable-pip-version-check

ENTRYPOINT ["streamlit"]
CMD ["run", "main.py"]