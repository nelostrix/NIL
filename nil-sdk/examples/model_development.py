"""Offline regression → checkpoint → registry → quantization → optional ONNX."""

import argparse
import json

import torch

import nil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="./nil_runs")
    parser.add_argument("--onnx", action="store_true")
    args = parser.parse_args()
    nil.runtime.seed(42)
    torch.set_num_threads(1)
    x = torch.linspace(-1, 1, 96).reshape(-1, 1)
    data = nil.data.tensors(x, x * 2 + 1)
    model = nil.nn.mlp(1, 1, hidden=())
    before = nil.evaluate(model, data, device_name="cpu")
    result = nil.train(
        model, data, epochs=40, learning_rate=0.05, device="cpu", output_dir=args.output
    )
    registry = nil.Registry(f"{args.output}/registry")
    registered = registry.register("regression", result.checkpoint, {"run": result.run.id})
    restored = registry.load("regression", nil.nn.mlp(1, 1, hidden=()))
    quantized = nil.optimization.quantize(restored)
    report = {
        "run": str(result.run.path),
        "before": before,
        "after": nil.evaluate(restored, data, device_name="cpu"),
        "quantized": nil.evaluate(quantized, data, device_name="cpu"),
        "artifact": registered,
    }
    if args.onnx:
        path = nil.optimization.export_onnx(
            restored,
            x[:2],
            result.run.path / "model.onnx",
            input_names=["features"],
            output_names=["prediction"],
            dynamic_axes={"features": {0: "batch"}, "prediction": {0: "batch"}},
        )
        report["onnx_prediction"] = nil.inference.OnnxModel(path).predict(x[:2].numpy())[0].tolist()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
