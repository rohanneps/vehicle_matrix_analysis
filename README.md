# Vehicle Data Management Framework

This is a utility project that aims to read, manipulate and analyze vehicle cordinates data.

## Author
+ Rohan Man Amatya


## Installation

#### Install python3

#### Venv

Run script
```
./create_venv.sh
```


## Usage

```
from framework import length, VehicleDataManager
obj = VehicleDataManager("./data.npy")

filtered_segment = obj.filter_by_id("4")  # or  obj.filter_by_id(4)
length_of_filtered_trajectory = obj.filter(length)
obj.plot(filtered_segment)

```
## Test
```
pytest -v tests/tests.py
