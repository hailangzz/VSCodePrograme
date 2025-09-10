
import onnxruntime as ort
import numpy as np

def verify_model(model_path):
    sess = ort.InferenceSession(model_path)
    input_name = sess.get_inputs()[0].name
    dummy_input = np.random.rand(1,3,640,640).astype(np.float32)
    
    outputs = sess.run(None, {input_name: dummy_input})
    print(f"Model {model_path} outputs shape:", [o.shape for o in outputs])

if __name__ == "__main__":
    verify_model("yolov8n_quant_static.onnx")
    verify_model("yolov8n_quant_dynamic.onnx")
