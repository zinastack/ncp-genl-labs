"""Ensemble step 3: softmax over the two SST-2 classes, return the winning label and its probability."""

import numpy as np
import triton_python_backend_utils as pb_utils

LABELS = np.array([b"NEGATIVE", b"POSITIVE"], dtype=np.object_)


class TritonPythonModel:
    def execute(self, requests):
        responses = []
        for request in requests:
            logits = pb_utils.get_input_tensor_by_name(request, "logits").as_numpy()  # (batch, 2)
            e = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs = e / e.sum(axis=1, keepdims=True)
            idx = probs.argmax(axis=1)
            responses.append(pb_utils.InferenceResponse(output_tensors=[
                pb_utils.Tensor("LABEL", LABELS[idx].reshape(-1, 1)),
                pb_utils.Tensor("SCORE", probs.max(axis=1).reshape(-1, 1).astype(np.float32)),
            ]))
        return responses
