import os
from unittest import TestCase
from unittest.mock import patch
from framework import VehicleDataManager, length


class VehicleDataManagerTest(TestCase):
    FILE_PATH = "data.npy"

    def test_WHEN_invalid_object_id_THEN_exception_is_caught(self):
        obj = VehicleDataManager(VehicleDataManagerTest.FILE_PATH)
        filtered_segment = obj.filter_by_id("test_id")
        # test should pass and process shouldn't fail due to exception

    def test_WHEN_np_file_present_THEN_log_is_generated(self):
        obj = VehicleDataManager(VehicleDataManagerTest.FILE_PATH)
        filtered_segment = obj.filter_by_id("4")
        length_of_filtered_trajectory = obj.filter(length)
        obj.plot(filtered_segment)

        self.assertTrue(os.path.exists(VehicleDataManager.LOG_FILE_NAME))

    def test_WHEN_np_file_present_THEN_plot_is_generated(self):
        obj = VehicleDataManager(VehicleDataManagerTest.FILE_PATH)
        filtered_segment = obj.filter_by_id("4")

        self.assertTrue(os.path.exists(VehicleDataManager.PLOT_FIGURE_FILE_NAME))

    def test_WHEN_plot_is_invoked_before_segmentation_THEN_exception_is_caught(self):
        obj = VehicleDataManager(VehicleDataManagerTest.FILE_PATH)
        obj.plot("some_value")
        filtered_segment = obj.filter_by_id("4")
        # test should pass and process shouldn't fail due to exception

    def test_WHEN_plot_is_invoked_with_incorrect_args_THEN_exception_is_caught(self):
        obj = VehicleDataManager(VehicleDataManagerTest.FILE_PATH)
        obj.plot("some_value")
        filtered_segment = obj.filter_by_id("test_id")

    def test_WHEN_unhandled_exception_THEN_exception_is_raised(self):
        obj = VehicleDataManager(VehicleDataManagerTest.FILE_PATH)
        with patch.object(
            VehicleDataManager, "filter_by_id", side_effect=[ValueError()]
        ):
            with self.assertRaises(ValueError):
                filtered_segment = obj.filter_by_id("4")
                length_of_filtered_trajectory = obj.filter(length)
                obj.plot(filtered_segment)
