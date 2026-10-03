# One image serves every model in model_repository/: the Python backends (text_classifier,
# tokenizer, postprocess) need torch + transformers; ONNX Runtime and TensorRT backends ship with Triton.
# Pin the Triton release; never deploy ":latest".
ARG TRITON_VERSION=25.08
FROM nvcr.io/nvidia/tritonserver:${TRITON_VERSION}-py3

ENV PIP_BREAK_SYSTEM_PACKAGES=1
RUN pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128 \
 && pip install --no-cache-dir "transformers>=4.44,<5" onnx

# Bake the weights into the image so pods start without internet access and cold starts are faster.
ENV HF_HOME=/opt/hf
RUN python -c "from transformers import pipeline; pipeline('text-classification', model='distilbert/distilbert-base-uncased-finetuned-sst-2-english')" \
 && chmod -R a+rX /opt/hf
ENV HF_HUB_OFFLINE=1

COPY model_repository /models
CMD ["tritonserver", "--model-repository=/models", "--model-control-mode=explicit", "--load-model=text_classifier"]
