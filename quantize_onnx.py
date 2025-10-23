import argparse, os
from onnxruntime.quantization import quantize_dynamic, QuantType
import onnx

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx_in", type=str, required=True)
    ap.add_argument("--onnx_out", type=str, required=True)
    args = ap.parse_args()

    # Load and clean the model
    model = onnx.load(args.onnx_in)
    # Remove value_info to avoid shape conflicts
    model.graph.ClearField("value_info")
    # Save cleaned model
    cleaned_path = args.onnx_in.replace('.onnx', '_cleaned.onnx')
    onnx.save(model, cleaned_path)

    quantize_dynamic(
        model_input=cleaned_path,
        model_output=args.onnx_out,
        weight_type=QuantType.QInt8,
    )
    kb = os.path.getsize(args.onnx_out)/1024
    print(f"Saved INT8 ONNX: {args.onnx_out} ({kb:.1f} KB)")

if __name__ == "__main__":
    main()
