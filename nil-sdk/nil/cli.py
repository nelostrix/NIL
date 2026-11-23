"""NIL SDK diagnostics and offline acceptance demo."""

import argparse
import json

from nil.runtime import capabilities


def main():
    # Redirected Windows streams can default to a legacy code page; the catalog
    # and bundled help use Unicode regardless of terminal/pipe configuration.
    import sys
    for stream in (sys.stdout,sys.stderr):
        reconfigure=getattr(stream,'reconfigure',None)
        if reconfigure is not None:reconfigure(encoding='utf-8',errors='replace')
    parser = argparse.ArgumentParser(prog="nil", description="NIL AI development SDK")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Report packages; does not load models or contact services")
    commands.add_parser("models", help="List models by group without loading backends")
    help_command = commands.add_parser("help", help="Read a module or model's bundled help.md")
    help_command.add_argument("topic", nargs="?", default="catalog")
    help_command.add_argument("--list", action="store_true", dest="list_topics")
    demo = commands.add_parser("demo", help="Train a tiny CPU regression model locally")
    demo.add_argument("--output", default="./nil_runs")
    args = parser.parse_args()
    if args.command == "doctor":
        print(json.dumps(capabilities(), indent=2))
    elif args.command == "help":
        from nil.documentation import read_help, topics

        try:
            print("\n".join(topics()) if args.list_topics else read_help(args.topic))
        except (ValueError, OSError) as error:
            parser.error(str(error))
    elif args.command == "models":
        from nil.catalog import tree

        print("NIL")
        groups = list(tree().items())
        for index, (group, names) in enumerate(groups):
            last = index == len(groups) - 1
            print(f"{'└──' if last else '├──'} {group}")
            for position, name in enumerate(names):
                prefix = "    " if last else "│   "
                branch = "└──" if position == len(names) - 1 else "├──"
                print(f"{prefix}{branch} {name}")
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
