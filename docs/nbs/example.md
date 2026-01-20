# Example

This example demonstrates how to use gsim with GDSFactory.

## A plot

Below you can see a plot:

```python
import matplotlib.pyplot as plt

plt.plot([0, 1])
plt.show()
```

## HTML Output

```python
from IPython.display import HTML

HTML("<strong style='color: red'>HEY!</strong>")
```

## Altair Example

```python
import altair as alt
import numpy as np
import pandas as pd

np.random.seed(42)
columns = ["A", "B", "C"]
source = pd.DataFrame(
    np.cumsum(np.random.randn(100, 3), 0).round(2),
    columns=columns,
    index=pd.RangeIndex(100, name="x"),
)
source = source.reset_index().melt("x", var_name="category", value_name="y")

# Create a selection that chooses the nearest point & selects based on x-value
nearest = alt.selection_point(nearest=True, on="pointerover", fields=["x"], empty=False)

# The basic line
line = (
    alt.Chart(source)
    .mark_line(interpolate="basis")
    .encode(x="x:Q", y="y:Q", color="category:N")
)

# Transparent selectors across the chart. This is what tells us
# the x-value of the cursor
selectors = (
    alt.Chart(source)
    .mark_point()
    .encode(
        x="x:Q",
        opacity=alt.value(0),
    )
    .add_params(nearest)
)
when_near = alt.when(nearest)

# Draw points on the line, and highlight based on selection
points = line.mark_point().encode(
    opacity=when_near.then(alt.value(1)).otherwise(alt.value(0))
)

# Draw text labels near the points, and highlight based on selection
text = line.mark_text(align="left", dx=5, dy=-5).encode(
    text=when_near.then("y:Q").otherwise(alt.value(" "))
)

# Draw a rule at the location of the selection
rules = (
    alt.Chart(source)
    .mark_rule(color="gray")
    .encode(
        x="x:Q",
    )
    .transform_filter(nearest)
)

# Put the five layers into a chart and bind the data
alt.layer(line, selectors, points, rules, text).properties(width=600, height=300)
```

## GDSFactory Component

```python
from gdsfactory.components import mzi

mzi()
```
