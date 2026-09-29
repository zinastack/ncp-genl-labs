# Triton + the Python dependencies the text_classifier Python backend needs.
# Pin the Triton release; never deploy ":latest".
ARG TRITON_VERSION=24.12
FROM nvcr.io/nvidia/tritonserver:${TRITON_VERSION}-py3

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124 \
 && pip install --no-cache-dir "transformers>=4.44,<5"

# Bake the weights into the image so pods start without internet access and cold starts are faster.
ENV HF_HOME=/opt/hf
RUN python -c "from transformers import pipeline; pipeline('text-classification', model='distilbert/distilbert-base-uncased-finetuned-sst-2-english')"

COPY model_repository /models
CMD ["tritonserver", "--model-repository=/models", "--strict-readiness=true"]
