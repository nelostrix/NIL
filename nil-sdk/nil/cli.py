"""NIL SDK diagnostics and offline acceptance demo."""

import argparse
import json

from nil.runtime import capabilities


def main():
    parser = argparse.ArgumentParser(prog="nil", description="NIL AI development SDK")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Report packages; does not load models or contact services")
    demo = commands.add_parser("demo", help="Train a tiny CPU regression model locally")
    demo.add_argument("--output", default="./nil_runs")
    args = parser.parse_args()
    if args.command == "doctor":
        print(json.dumps(capabilities(), indent=2))
    else:
        import nil
        from nil.runtime import require, seed

        torch = require("torch", "torch")
        seed(7)
        x = torch.linspace(-1, 1, 64).reshape(-1, 1)
        data = nil.data.tensors(x, 3 * x - 0.5)
        model = nil.nn.mlp(1, 1, hidden=())
        before = nil.evaluate(model, data, device_name="cpu")
        result = nil.train(
            model,
            data,
            epochs=30,
            batch_size=16,
            learning_rate=0.05,
            device="cpu",
            output_dir=args.output,
        )
        after = nil.evaluate(model, data, device_name="cpu")
        if after["mse"] >= before["mse"]:
            raise RuntimeError("Training acceptance failed: loss did not improve")
        print(
            json.dumps(
                {
                    "status": result.status,
                    "run": str(result.run.path),
                    "before": before,
                    "after": after,
                    "checkpoint": str(result.checkpoint),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
