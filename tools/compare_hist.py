#pip install opencv-python

import cv2
import numpy as np

def read_histogram_from_file(file_name):
    hists = {}
    with open(file_name) as f:
        name_line = None
        range_line = None
        buket_line = None
        count_line = None
        while (line := f.readline().rstrip()):
            if line.startswith("Arg Name:"):
                if name_line:
                    is_valid = name_line \
                        and (range_line and "nan number count" not in range_line) \
                        and buket_line \
                        and (count_line and "All values are" not in count_line)
                    if is_valid:
                        # do parsing
                        name, shape_str = name_line.strip().removeprefix("Arg Name:").strip().split()
                        range_str = range_line.strip().removeprefix("Value range").strip()
                        bucket_str_array = buket_line.strip().split()
                        count_str_array = count_line.strip().split()
                        if len(bucket_str_array) == len(count_str_array) + 1:
                            print("valid pair ", name, shape_str, range_str, bucket_str_array, count_str_array, len(bucket_str_array), len(count_str_array))
                            h = np.histogram([float(v) for v in count_str_array], bins=[float(b) for b in bucket_str_array])
                            hists[name] = [name, shape_str, h]
                        else:
                            pass
                            # print("invalid pair ", name, shape_str, range_str, bucket_str_array, count_str_array, len(bucket_str_array), len(count_str_array))
                        range_line = None
                        buket_line = None
                        count_line = None
                        name_line = line
                    else:
                        # print("ignoring invalid pair ", name_line, range_line, buket_line, count_line)
                        range_line = None
                        buket_line = None
                        count_line = None
                        name_line = line
                else:
                    name_line = line
                    assert not range_line and not count_line
            elif range_line is None:
                range_line = line
            elif buket_line is None:
                buket_line = line
            elif count_line is None:
                count_line = line
    return hists





hists1 = read_histogram_from_file("../d/rocm_fp16_scale2/iter_0_stat.out")
hists2 = read_histogram_from_file("../d/cuda_fp16_scale2/iter_0_stat.out")

shared_names = [n for n in hists1 if n in hists2]
# c = np.histogram([1, 2, 1], bins=[0, 1, 2, 3])
with open("./hist_diff.txt", 'w') as f:
    for n in shared_names:
        a = hists1[n][2]
        c = hists2[n][2]
        score = cv2.compareHist(a[0].ravel().astype('float32'), c[0].ravel().astype('float32'), cv2.HISTCMP_BHATTACHARYYA)
        print(n, score)
        f.write(n + " " + str(score) + "\n")