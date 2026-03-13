FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir wyoming httpx aiohttp
COPY wyoming_chatterbox.py .
EXPOSE 10201 10202
CMD ["python", "wyoming_chatterbox.py"]
