# Base image for TorchServe
FROM pytorch/torchserve:latest-cpu

WORKDIR /home/model-server


# Install dependencies
COPY requirements.txt  /home/model-server/
RUN pip install --no-cache-dir -r /home/model-server/requirements.txt

COPY model_store/lipnet.mar /home/model-server/model_store/
COPY src/model_handler.py /home/model-server/

# Expose the port
EXPOSE 8080 8081 8082

# Start the model server
CMD ["torchserve", "--start", "--model-store", "model_store", "--models", "lipnet.mar", "--dt", "--ncs"]

