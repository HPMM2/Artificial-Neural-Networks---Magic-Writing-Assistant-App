from onnx2tflite import onnx_converter

onnx_converter(
    onnx_model_path = "./letter_model.onnx",        # ← cambia si tu modelo se llama distinto
    need_simplify = True,
    output_path = "./modelos_convertidos",   # ← puedes cambiar esta carpeta
    target_formats = ['tflite']              # o ['keras', 'tflite'] si quieres ambos
)
