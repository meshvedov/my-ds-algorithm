import numpy as np
import pandas as pd
from pprint import pprint


class MyTreeReg:
    def __init__(self,
                 max_depth=5,
                 min_samples_split=2,
                 max_leafs=20) -> None:
        self.max_leafs = max_leafs
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__} class: max_depth={self.max_depth}, min_samples_split={self.min_samples_split}, "
            f"max_leafs={self.max_leafs}"
        )


mtr = MyTreeReg()
print(mtr)
