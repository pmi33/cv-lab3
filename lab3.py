import cv2
import numpy as np


def color_rank_filter(image: np.ndarray, mask:np.ndarray, rank: int):
    """
    Разбиваем изображение на 3 канала RGB
    и применяем фильтр к каждому каналу
    :param image: numpy-массив с пикселями изображения (размер ВЫСОТА х ШИРИНА х 3)
    :param mask: маска фильтра (двумерный numpy-массив)
    :param rank: ранг
    :return: отфильтрованное изображение
    """
    res = np.zeros_like(image)
    for i in range(3):
        res[..., i] = rank_filter(image[..., i], mask, rank)
    return res


def rank_filter(pixels: np.ndarray, mask:np.ndarray, rank: int):
    """
    Ранговая фильтрация конкретного канала.
    До фильтрации изображение увеличивается (чтобы можно было края обработать).
    :param pixels: numpy-массив с пикселями одного канала изображения
    :param mask: маска фильтра (двумерный numpy-массив)
    :param rank: ранг
    :return: отфильтрованное изображение
    """
    h, w = pixels.shape
    offset = mask.shape[0] // 2
    res = np.zeros_like(pixels)
    expanded = expand(pixels, mask)
    for i in range(offset, h + offset):
        for j in range(offset, w + offset):
            frame = expanded[i - offset : i + offset + 1, j - offset : j + offset + 1]
            # делаем из квадратного куска одномерный массив
            ravel = frame.ravel()
            # получаем список индексов элементов одномерного массива, который мы будем сортировать
            index = get_indexes(mask)
            # сортируем
            sorted_array = np.sort(ravel[index])
            res[i - offset, j - offset] = sorted_array[rank]
    return res


def expand(pixels, mask):
    """
    Увеличение исходного изображения.
    ВАЖНО: первый индекс
    :param pixels: numpy-массив с пикселями
    :param mask: маска
    :return: изображение, увеличенное с каждой стороны на половину размера маски
    """
    h, w = pixels.shape
    offset = mask.shape[0] // 2
    expanded = np.zeros(shape=(h + 2 * offset, w + 2 * offset))
    # заполняем центр нового (увеличенного) изображения
    expanded[offset: h + offset, offset: w + offset] = pixels
    # заполнение краев увеличенного изображения
    for i in range(offset):
        expanded[offset: h + offset, i] = pixels[:, 0]
        expanded[offset: h + offset, w + offset + i] = pixels[:, w - 1]
    for i in range(offset):
        expanded[i, :] = expanded[offset, :]
        expanded[h + offset + i, :] = expanded[h + offset - 1, :]
    return expanded


def get_indexes(mask: np.ndarray):
    """
    Получение индексов элементов маски
    :param mask: маска
    :return: массив с индексами, например, для маски из единиц
             размером 3х3 будет возвращен массив [0,1,2,3,4,5,6,7,8]
    """
    ravel = mask.ravel()
    index = np.array([])
    # i - индекс, val - значение
    # если val=3, то i 3 раза встретится в генерируемом массиве
    for i, val in enumerate(ravel):
        index = np.append(index, np.array([i] * val))
    return index.astype(np.int32)

if __name__ == '__main__':
    img = cv2.imread("img3.jpg", cv2.IMREAD_COLOR)

    mask = np.ones((3, 3)).astype(np.int32)
    filtered = color_rank_filter(img, mask, rank=5)

    cv2.imwrite("filtered.jpg", filtered)