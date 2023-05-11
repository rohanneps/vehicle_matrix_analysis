import os
from typing import Any, Callable, Union
from matplotlib import pyplot as plt
import numpy as np
from numpy import ndarray
from .utility import length


class VehicleDataManager:
    """
    Handle operations on vehicle data
    """

    PLOT_FIGURE_FILE_NAME: str = os.path.join(".", "plot.png")
    DIM_INDEX: int = 0
    DIM_OBJECT_ID: int = 1
    DIM_LATITUDE: int = 2
    DIM_LONGITUDE: int = 3
    OBJECT_ID_OCCURANCE_FILTER_THRESHOLD: int = 2

    def catch_exception(func):
        def wrapper(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except Exception as e:  # work on python 3.x
                print(e)

        return wrapper

    def __init__(self, np_file_path: str):
        self._file_path: str = np_file_path
        self._filtered_segment: ndarray
        self._ids_to_hold: ndarray
        self._has_error: bool = False
        self._np_data: ndarray

        self._validate_file_path()
        self._load_file_content()
        self._preprocess()

    @catch_exception
    def filter_by_id(self, object_id: Union[int, str]) -> ndarray:
        """
        Return a subset of data for the given id
        """
        try:
            object_id = int(object_id)
        except ValueError:
            error = f"{object_id} is not a valid object_id "
            raise TypeError(error)

        self._filtered_segment = self._np_data[
            self._np_data[:, self.DIM_OBJECT_ID] == object_id
        ]
        return self._filtered_segment

    @catch_exception
    def filter(self, func: Callable) -> Any:
        """
        Return the output of callable on filtered segment
        """
        self._validate_filtered_segment()
        return func(self._filtered_segment)

    @catch_exception
    def plot(self, trajectory: ndarray) -> None:
        """
        plot and save the trajectory
        """
        self._validate_filtered_segment()
        plt.figure().clear()
        plt.title(f"Trajectory for object_id: {trajectory[:,1][0]}")
        plt.xlabel("LATITUDE")
        plt.ylabel("LONGITUDE")
        plt.plot(trajectory[:, self.DIM_LATITUDE], trajectory[:, self.DIM_LONGITUDE])
        plt.savefig(self.PLOT_FIGURE_FILE_NAME)
        print(f"Plot has been saved to {self.PLOT_FIGURE_FILE_NAME}")

    @catch_exception
    def _load_file_content(self) -> None:
        """
        load file content
        """
        if not self._has_error:
            self._np_data = np.load(self._file_path)

    @catch_exception
    def _preprocess(self):
        """
        sort the numpy data in timestamp index id and filter unique object_ids
        """
        if self._has_error:
            return

        # sort index: 1) Sort by index 2) arrange object_id, lat, long by the sorted index
        self._np_data = self._np_data[
            np.argsort(self._np_data, axis=self.DIM_INDEX)[:, self.DIM_INDEX]
        ]

        # filter single occurance object_ids
        unique_elements, counts_elements = np.unique(
            self._np_data[:, self.DIM_OBJECT_ID], return_counts=True
        )
        self._ids_to_hold = unique_elements[
            counts_elements >= self.OBJECT_ID_OCCURANCE_FILTER_THRESHOLD
        ]
        self._np_data = self._np_data[
            np.in1d(self._np_data[:, self.DIM_OBJECT_ID], self._ids_to_hold)
        ]

    def _validate_file_path(self) -> None:
        """
        Check if the file exists
        """
        if not os.path.exists(self._file_path):
            self._has_error = True
            print(f"Sorry, {self._file_path} not found")

    def _validate_filtered_segment(self) -> None:
        """
        Raise exception if filtered segment is not set
        """
        if getattr(self, "_filtered_segment", None) is None:
            raise Exception("Please create a trajectory using filter_by_id method")


if __name__ == "__main__":
    obj = VehicleDataManager("./data.npy")

    filtered_segment = obj.filter_by_id("4")  # or  obj.filter_by_id(4)
    length_of_filtered_trajectory = obj.filter(length)
    obj.plot(filtered_segment)
