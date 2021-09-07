FROM tiangolo/uvicorn-gunicorn-fastapi

COPY ./app /app

ENV PORT=8080
EXPOSE 8080

COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt \
        --disable-pip-version-check \
        --no-cache-dir