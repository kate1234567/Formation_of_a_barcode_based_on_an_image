import os
import math
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from barcode import EAN8
from barcode.writer import ImageWriter
from scipy import signal
import cv2 as cv

def corr(img1, img2):
 #преобразование изображений в формат numpy array
 img1 = np.array(img1)
 img2 = np.array(img2)

 #преобразование в формат float и масштабирование значений в диапазон от 0 до 1
 img1 = img1.astype(float) / 255.0
 img2 = img2.astype(float) / 255.0

 #вычисление корреляции
 corr = np.corrcoef(img1.flatten(), img2.flatten())[0, 1]
 return corr

def ssim(img1, img2, k1=0.01, k2=0.03, L=255):
    # Конвертируем изображения в тип данных float
    img1 = img1.astype(float)
    img2 = img2.astype(float)

    # Константы для вычисления SSIM
    c1 = (k1*L)**2
    c2 = (k2*L)**2
    # Окно Гаусса для оценки SSIM на локальном уровне изображения (для вычисления средних и ковариаций)
    # Общепринятое значение - 11x11
    gauss_window = np.ones((11, 11))

    # Вычисляем средние значения
    mu1 = signal.convolve2d(img1, gauss_window, mode='valid') / np.sum(gauss_window)
    mu2 = signal.convolve2d(img2, gauss_window, mode='valid') / np.sum(gauss_window)

    # Вычисляем дисперсии
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2

    # Вычисляем ковариации
    sigma1_sq = signal.convolve2d(img1**2, gauss_window, mode='valid') / np.sum(gauss_window) - mu1_sq
    sigma2_sq = signal.convolve2d(img2**2, gauss_window, mode='valid') / np.sum(gauss_window) - mu2_sq
    sigma1_sigma2 = signal.convolve2d(img1*img2, gauss_window, mode='valid') / np.sum(gauss_window) - mu1_mu2

    # Вычисляем значение SSIM
    numerator = (2*mu1_mu2 + c1) * (2*sigma1_sigma2 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    ssim_map = numerator / denominator
    return np.mean(ssim_map)

def crop_img(image):
    image_arr = np.array(image)
    numrows = len(image_arr)
    top = int(numrows * 0.1)
    bottom = int(numrows * 0.69)
    image_arr = image_arr[top:bottom]
    return image_arr

def numbers(image_arr):
    distance = []
    for i in range(8):
        distance.append([])
    for i in range(8):
        numrows = len(image_arr)
        bottom1 = numrows * (i + 1) // 9
        bottom2 = numrows * (i + 2) // 9
        area1 = image_arr[i:bottom1]
        area2 = image_arr[bottom1:bottom2]
        grad1 = np.gradient(area1.max(axis=1))
        grad2 = np.gradient(area2.max(axis=1))
        result = 0

        min_length = min(len(grad1), len(grad2))

        for j in range(min_length):
            result += (grad2[j] - grad1[j]) ** 2
        distance[i] = math.sqrt(result)
    return distance

def ean8_checksum(digits):
    weights = [3, 1, 3, 1, 3, 1, 3]
    total = sum(w * int(d) for w, d in zip(weights, digits))
    checksum = 10 - (total % 10)
    return 0 if checksum == 10 else checksum

def barcode_generation():
    root_folder = "faces94"
    folders = os.listdir(root_folder)
    barcodes_folder = "barcodes"
    hist = []

    if not os.path.exists(barcodes_folder):
        os.makedirs(barcodes_folder)

    for folder in folders:
        folder_path = os.path.join(root_folder, folder)
        images = os.listdir(folder_path)
        distances = []

        for image_file in images:
          img_path = os.path.join(folder_path, image_file)
          img = Image.open(img_path).convert("L")
          img = crop_img(img)
          distance = numbers(img)
          distances.append(distance)

        # Вычисление среднего значения расстояний для всех изображений в папке
        mean_distance = np.mean(distances, axis=0)

        # Нормализация и преобразование средних расстояний в числа для штрих-кода
        normalized_distances = (mean_distance - np.min(mean_distance)) / (np.max(mean_distance) - np.min(mean_distance))
        barcode_numbers = [round(x * 9) for x in normalized_distances]

        # Берем первые 7 чисел и добавляем контрольную сумму
        barcode_numbers = barcode_numbers[:7]
        barcode_numbers.append(ean8_checksum(barcode_numbers))

        hist. append(barcode_numbers)

        # Генерация и сохранение штрих-кода
        barcode_digits = "".join([str(d) for d in barcode_numbers])
        barcode = EAN8(barcode_digits, writer=ImageWriter())

        # Создание папки для каждого человека в каталоге barcodes
        person_barcodes_folder = os.path.join(barcodes_folder, folder)
        if not os.path.exists(person_barcodes_folder):
            os.makedirs(person_barcodes_folder)

       # Сохранение штрих-кода для каждого класса
        for idx in range(10):
           barcode.save(os.path.join(person_barcodes_folder, f"{idx}"))

    return hist

def import_img(i, j, type):
    if type == 'img':
        image = cv.imread('faces94/s' + str(i + 1) + '/' + str(j + 1) + '.jpg')
    if type == 'barcode':
        image = cv.imread('barcodes/s' + str(i + 1) + '/' + str(j + 1) + '.png')
    if type == 'ssim':
        image = cv.imread('faces94/s' + str(i + 1) + '/' + str(j + 1) + '.jpg', 0)
    return image

def show_results():
    hist = []
    hist = barcode_generation()
    plt.figure(figsize=(10, 5))
    barcode_generation()

    for i in range(35):
        corr_g = []
        ssim_g = []
        for j in range(10):
            ax1 = plt.subplot(3, 2, 1)
            ax2 = plt.subplot(3, 2, 2)
            ax3 = plt.subplot(3, 2, 3)
            ax4 = plt.subplot(3, 2, 5)
            ax5 = plt.subplot(3, 2, 6)

            ax1.clear()
            ax1.imshow(import_img(i, j, 'img'))
            ax1.set_xticks([])
            ax1.set_yticks([])
            ax1.set_xlabel('Класс № ' + str(i + 1) + ' Изображение № ' + str(j + 1))

            ax2.clear()
            ax2.hist(hist[i])

            ax3.clear()
            ax3.set_xticks([])
            ax3.set_yticks([])
            ax3.imshow(import_img(i, j - 1, 'barcode'))

            if j < 9:
                corr_ = corr(import_img(0, j, 'ssim'), import_img(0, j + 1, 'ssim'))
                ssim_ = ssim(import_img(0, j, 'ssim'), import_img(0, j + 1, 'ssim'))
                corr_g.append(corr_)
                ssim_g.append(ssim_)
                ax4.clear()
                ax5.clear()
                corr_text = 'Корреляция для 1 и ' + str(j + 2) + ' изображений = ' + str(corr_)
                ssim_text = 'SSIM-индекс для 1 и ' + str(j + 2) + ' изображений = ' + str(ssim_)
                ax4.text(0.02, 0.5, corr_text)
                ax4.set_xticks([])
                ax4.set_yticks([])
                ax5.text(0.02, 0.5, ssim_text)
                ax5.set_xticks([])
                ax5.set_yticks([])

            plt.pause(1)

        plt.figure(figsize=(10, 5))
        axcorr = plt.subplot(1, 2, 1)
        axssim = plt.subplot(1, 2, 2)

        axcorr.plot(corr_g)
        axcorr.set_xticks([])
        axcorr.set_xlabel('Количество изображений в одном классе')
        axcorr.set_ylabel('Корреляция')

        axssim.plot(ssim_g)
        axssim.set_xticks([])
        axssim.set_xlabel('Количество изображений в одном классе')
        axssim.set_ylabel('SSIM-индекс')

        plt.show()

show_results()