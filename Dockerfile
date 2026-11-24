# Used by Hugging Face Spaces (or any Docker host) to run Neurova online.
FROM python:3.12-slim
WORKDIR /neurova
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Hosting services give the container a read-only code folder, so the database goes in /tmp.
ENV PORT=7860 NEUROVA_DATA=/tmp/neurova-data
EXPOSE 7860
CMD ["python", "run.py"]
