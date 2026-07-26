import unittest

from training.preprocessing.create_segmentation_annotation_set import select_diverse


class SegmentationAnnotationSelectionTests(unittest.TestCase):
    def test_diverse_selection_is_unique_and_deterministic(self):
        rows = [
            {
                "parent_sample_id": f"sample-{index}",
                "mask_area_ratio": str(0.4 + index * 0.01),
                "bbox_xyxy": f"{index},{index},{100 + index},{200 + index}",
            }
            for index in range(20)
        ]

        first = select_diverse(rows, 5)
        second = select_diverse(rows, 5)

        self.assertEqual(first, second)
        self.assertEqual(len({row["parent_sample_id"] for row in first}), 5)

    def test_selection_rejects_oversampling(self):
        row = {
            "parent_sample_id": "sample",
            "mask_area_ratio": "0.5",
            "bbox_xyxy": "0,0,100,200",
        }
        with self.assertRaisesRegex(ValueError, "Requested"):
            select_diverse([row], 2)


if __name__ == "__main__":
    unittest.main()
