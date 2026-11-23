# catalog

Discover models and register extensions through a shallow catalog.

## Usage

```python
import nil

print(nil.list_models("regression"))
print(nil.catalog.tree())
print(nil.help("linear_regression"))
```

## Parameters

`group`: optional category. `nil.model(name, **parameters)` constructs a native model.

## Output

Model names, a group-to-names dictionary, or the selected native model.

## Requirements

Core SDK only for discovery/help. Choose an optional compute backend to construct a model.

## Limitations

Listing adapters does not verify installations or hardware. nil.fit handles native classical estimators/pipelines; nil.train handles neural modules. nil.evaluate and nil.predict dispatch between them.

## Add a module

```python
import nil

def factory(**parameters):
    from my_package import MyModel
    return MyModel(**parameters)

nil.register_model("my_model", factory, group="regression", help_path="my_model/help.md")
```

Keep dependencies inside the factory. Registration does not construct the model. Pass an absolute help_path when the working directory can change. Registration is explicit and process-local; import your module in each process that uses it. Adding a constructor does not automatically add training/evaluation/serialization for an unfamiliar backend.
