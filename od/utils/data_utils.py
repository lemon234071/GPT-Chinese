import pickle
import json
import matplotlib.pyplot as plt


def load_pkl(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def load_txt(path):
    with open(path, encoding='UTF-8', errors='ignore') as f:
        data = [i.strip() for i in f.readlines() if len(i) > 0]
    return data


def save_txt(data, path):
    with open(path, 'w', encoding='UTF-8') as f:
        f.write(data)


def load_json(path):
    with open(path, 'r', encoding='UTF_8') as f:
        return json.load(f)


def save_json(data, path, indent=0):
    with open(path, 'w', encoding='UTF-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def plt_xy(path, x, y):
    # 创建一个点数为 8 x 6 的窗口, 并设置分辨率为 80像素/每英寸
    plt.figure()
    # 再创建一个规格为 1 x 1 的子图
    # plt.subplot(111)
    plt.plot(x, y)
    #plt.show()
    plt.savefig(path)
