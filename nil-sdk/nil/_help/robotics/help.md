# robotics

Build offline neural policies or explicitly use existing robotics applications.

## Usage

```python
import nil

policy = nil.robotics.policy(observation_dim=4, action_dim=2)
print(type(policy).__name__)
```

## Parameters

Policy observation/action dimensions; demonstrations require matching finite observation/action matrices.

## Output

Native neural policy or validated offline training data.

## Requirements

torch for policy training; robotics extra for physical-world helpers.

## Limitations

Creating a policy does not actuate hardware. Position/velocity filtering is not physical safety certification; device integrations need separate validation.
