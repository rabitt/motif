
# def array_equal(array1, array2):
#     return np.all(np.isclose(array1, array2))


# class TestContoursInit(unittest.TestCase):

#     def setUp(self):
#         self.path = os.path.abspath('tests/data/short.wav')
#         self.ctr = core.Contours(
#             self.path, 'salamon', recompute=False, clean=False
#         )

#     def test_audio_fpath(self):
#         expected = self.path
#         actual = self.ctr.audio_fpath
#         self.assertEqual(expected, actual)

#     def test_method(self):
#         expected = 'salamon'
#         actual = self.ctr.method
#         self.assertEqual(expected, actual)

#     def test_recompute(self):
#         expected = False
#         actual = self.ctr.recompute
#         self.assertEqual(expected, actual)

#     def test_clean(self):
#         expected = False
#         actual = self.ctr.clean
#         self.assertEqual(expected, actual)

#     def test_nums(self):
#         expected = set([0, 1, 2, 3, 4, 5, 6, 7, 8])
#         actual = self.ctr.nums
#         self.assertEqual(expected, actual)

#     def test_index_mapping(self):
#         expected = {
#             0: range(0, 129),
#             1: range(129, 325),
#             2: range(325, 697),
#             3: range(697, 853),
#             4: range(853, 989),
#             5: range(989, 1068),
#             6: range(1068, 1387),
#             7: range(1387, 1517),
#             8: range(1517, 1666)
#         }
#         actual = self.ctr.index_mapping
#         self.assertEqual(expected.keys(), actual.keys())
#         for k in expected.keys():
#             self.assertTrue(array_equal(expected[k], actual[k]))

#     def test_index(self):
#         expected = np.array(
#             [0]*129 +
#             [1]*196 +
#             [2]*372 +
#             [3]*156 +
#             [4]*136 +
#             [5]*79 +
#             [6]*319 +
#             [7]*130 +
#             [8]*149
#         )
#         actual = self.ctr.index
#         print sum(expected == 3)
#         print sum(actual == 3)
#         self.assertTrue(array_equal(expected, actual))

#     def test_times(self):
#         expected = np.array([0.325079, 0.327982, 0.330884, 0.333787, 0.336689])
#         actual = self.ctr.times[:5]
#         print actual
#         self.assertTrue(array_equal(expected, actual))

#     def test_freqs(self):
#         expected = np.array([198.275, 198.275, 199.424, 198.275, 198.275])
#         actual = self.ctr.freqs[:5]
#         print actual
#         self.assertTrue(array_equal(expected, actual))

#     def test_salience(self):
#         expected = np.array(
#             [0.44957045, 0.41604075, 0.44865103, 0.42353233, 0.48934882]
#         )
#         actual = self.ctr.salience[:5]
#         self.assertTrue(array_equal(expected, actual))

#     def test_sal_normalization(self):
#         expected = 1.0
#         actual = np.max(self.ctr.salience)
#         self.assertEqual(expected, actual)

#     def test_features(self):
#         expected = None
#         actual = self.ctr.features
#         self.assertEqual(expected, actual)

#     def test_labels(self):
#         expected = None
#         actual = self.ctr.labels
#         self.assertEqual(expected, actual)

#     def test_overlaps(self):
#         expected = None
#         actual = self.ctr.overlaps
#         self.assertEqual(expected, actual)

#     def test_scores(self):
#         expected = None
#         actual = self.ctr.scores
#         self.assertEqual(expected, actual)


# class TestContoursMethods(unittest.TestCase):

#     def setUp(self):
#         self.path = os.path.abspath('tests/data/short.wav')
#         self.ctr = core.Contours(
#             self.path, 'salamon', recompute=False, clean=False
#         )

#     def test_contour_times(self):
#         expected = np.array([0.325079, 0.327982, 0.330884, 0.333787, 0.336689])
#         actual_times = self.ctr.contour_times(0)
#         actual = actual_times[:5]
#         self.assertTrue(array_equal(expected, actual))

#         expected_length = 129
#         actual_length = len(actual_times)
#         self.assertEqual(expected_length, actual_length)

#     def test_contour_freqs(self):
#         expected = np.array([198.275, 198.275, 199.424, 198.275, 198.275])
#         actual_freqs = self.ctr.contour_freqs(0)
#         actual = actual_freqs[:5]
#         self.assertTrue(array_equal(expected, actual))

#         expected_length = 129
#         actual_length = len(actual_freqs)
#         self.assertEqual(expected_length, actual_length)

#     def test_contour_salience(self):
#         expected = np.array(
#             [0.44957045, 0.41604075, 0.44865103, 0.42353233, 0.48934882]
#         )
#         actual_salience = self.ctr.contour_salience(0)
#         actual = actual_salience[:5]
#         self.assertTrue(array_equal(expected, actual))

#         expected_length = 129
#         actual_length = len(actual_salience)
#         self.assertEqual(expected_length, actual_length)

#     def test_get_labels(self):
#         expected_labels = {
#             0: 1,
#             1: 0,
#             2: 1,
#             3: 0,
#             4: 0,
#             5: 0,
#             6: 1,
#             7: 0,
#             8: 0
#         }
#         expected_overlaps = {
#             0: 1.0,
#             1: 0,
#             2: 1.0,
#             3: 0,
#             4: 0,
#             5: 0,
#             6: 0.55339805825242716,
#             7: 0,
#             8: 0
#         }
#         self.ctr.get_labels('tests/data/short_annotation.csv')
#         actual_labels = self.ctr.labels
#         actual_overlaps = self.ctr.overlaps
#         self.assertEqual(expected_labels, actual_labels)
#         self.assertEqual(expected_overlaps, actual_overlaps)

#     def test_get_labels_olap(self):
#         expected_labels = {
#             0: 1,
#             1: 0,
#             2: 1,
#             3: 0,
#             4: 0,
#             5: 0,
#             6: 0,
#             7: 0,
#             8: 0
#         }
#         expected_overlaps = {
#             0: 1.0,
#             1: 0,
#             2: 1.0,
#             3: 0,
#             4: 0,
#             5: 0,
#             6: 0.55339805825242716,
#             7: 0,
#             8: 0
#         }
#         self.ctr.get_labels(
#             'tests/data/short_annotation.csv', overlap_threshold=0.6
#         )
#         actual_labels = self.ctr.labels
#         actual_overlaps = self.ctr.overlaps
#         self.assertEqual(expected_labels, actual_labels)
#         self.assertEqual(expected_overlaps, actual_overlaps)

#     def test_compute_features(self):
#         self.ctr.compute_features()
#         expected = np.array([
#             9.72662505e-01, 4.64411679e-04, 5.65660464e-02, 7.37664894e-01,
#             1.95228333e-01, 4.89478291e-01, 1.09299463e+01, 3.85710124e+01,
#             8.75000000e-01, 3.71520000e-01, 1.96272752e+02, 1.00511299e+00,
#             6.87200000e+00
#         ])
#         actual = self.ctr.features[0]
#         self.assertTrue(array_equal(expected, actual))

#         expected_keys = [0, 1, 2, 3, 4, 5, 6, 7, 8]
#         acutal_keys = self.ctr.features.keys()
#         self.assertEqual(expected_keys, acutal_keys)

#     def test_save(self):
#         self.ctr.save('tests/data/temp.csv')
#         os.remove('tests/data/temp.csv')

