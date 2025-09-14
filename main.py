import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import csv
from datetime import datetime

def load_image(image_path):
    """
    Загрузка изображения с проверкой ошибок
    """
    if not os.path.exists(image_path):
        print(f"Ошибка: Файл '{image_path}' не найден")
        return None
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Ошибка: Не удалось загрузить изображение '{image_path}'")
        return None
    
    print(f"Изображение загружено: {image_path}")
    return image

def convert_to_hsv(image):
    """
    Преобразование BGR в HSV
    """
    try:
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    except Exception as e:
        print(f"Ошибка преобразования в HSV: {e}")
        return None

def extract_hsv_channels(hsv_image):
    """
    Разделение HSV на каналы
    """
    try:
        return cv2.split(hsv_image)
    except Exception as e:
        print(f"Ошибка разделения каналов: {e}")
        return None, None, None

def detect_color(hsv_image, original_image, lower_bound, upper_bound):
    """
    Выделение цвета по диапазону HSV
    """
    try:
        mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
        result = cv2.bitwise_and(original_image, original_image, mask=mask)
        return mask, result
    except Exception as e:
        print(f"Ошибка выделения цвета: {e}")
        return None, None

def visualize_results(original, h, s, v, mask, result, color_name):
    """
    Визуализация результатов обработки
    """
    try:
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle('Анализ HSV цветового пространства и выделение голубого цвета', fontsize=16, y=0.98)
        
        plt.subplots_adjust(wspace=0.3, hspace=0.4)
        
        axes[0, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Оригинальное изображение (RGB)', fontsize=12, pad=10)
        axes[0, 0].axis('off')
        
        im1 = axes[0, 1].imshow(h, cmap='gray')
        axes[0, 1].set_title('Канал H (Hue)\nЦветовой тон', fontsize=12, pad=10)
        axes[0, 1].axis('off')
        
        im2 = axes[0, 2].imshow(s, cmap='gray')
        axes[0, 2].set_title('Канал S (Saturation)\nНасыщенность', fontsize=12, pad=10)
        axes[0, 2].axis('off')
        
        im3 = axes[1, 0].imshow(v, cmap='gray')
        axes[1, 0].set_title('Канал V (Value)\nЯркость', fontsize=12, pad=10)
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(mask, cmap='gray')
        axes[1, 1].set_title('Маска голубого цвета', fontsize=12, pad=10)
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        axes[1, 2].set_title('Выделенные голубые объекты', fontsize=12, pad=10)
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Ошибка визуализации: {e}")

def save_to_csv(image_path, image_shape, h_stats, s_stats, v_stats, cyan_pixels, lower_cyan, upper_cyan):
    """
    Сохранение статистики в CSV файл
    """
    try:
        filename = f"image_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            writer.writerow(['Параметр', 'Значение'])
            writer.writerow(['Путь к изображению', image_path])
            writer.writerow(['Размер изображения', f"{image_shape[1]}x{image_shape[0]}"])
            writer.writerow(['Количество каналов', image_shape[2]])
            writer.writerow(['Канал H - минимальное значение', h_stats['min']])
            writer.writerow(['Канал H - максимальное значение', h_stats['max']])
            writer.writerow(['Канал H - среднее значение', f"{h_stats['mean']:.1f}"])
            writer.writerow(['Канал S - минимальное значение', s_stats['min']])
            writer.writerow(['Канал S - максимальное значение', s_stats['max']])
            writer.writerow(['Канал S - среднее значение', f"{s_stats['mean']:.1f}"])
            writer.writerow(['Канал V - минимальное значение', v_stats['min']])
            writer.writerow(['Канал V - максимальное значение', v_stats['max']])
            writer.writerow(['Канал V - среднее значение', f"{v_stats['mean']:.1f}"])
            writer.writerow(['Диапазон H для голубого', f"{lower_cyan[0]}-{upper_cyan[0]}°"])
            writer.writerow(['Диапазон S для голубого', f"{lower_cyan[1]}-255"])
            writer.writerow(['Диапазон V для голубого', f"{lower_cyan[2]}-255"])
            writer.writerow(['Найдено голубых пикселей', cyan_pixels])
            writer.writerow(['Дата анализа', datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
        
        print(f"Результаты сохранены в файл: {filename}")
        return True
        
    except Exception as e:
        print(f"Ошибка при сохранении в CSV: {e}")
        return False

def main():
    """
    Основная функция программы
    """
    image_path = input("Введите путь к изображению: ")
    
    # 1. Загрузка изображения
    image = load_image(image_path)
    if image is None:
        return False
    
    # 2. Преобразование в HSV
    hsv_image = convert_to_hsv(image)
    if hsv_image is None:
        return False
    
    # 3. Разделение каналов
    h, s, v = extract_hsv_channels(hsv_image)
    if h is None:
        return False
    
    # 4. Выделение голубого цвета
    lower_cyan = np.array([90, 100, 100])
    upper_cyan = np.array([105, 255, 255])
    
    cyan_mask, cyan_result = detect_color(hsv_image, image, lower_cyan, upper_cyan)
    if cyan_mask is None:
        return False
    
    # 5. Визуализация результатов
    visualize_results(image, h, s, v, cyan_mask, cyan_result, "голубого")
    
    # 6. Сохранение статистики в CSV
    h_stats = {'min': h.min(), 'max': h.max(), 'mean': h.mean()}
    s_stats = {'min': s.min(), 'max': s.max(), 'mean': s.mean()}
    v_stats = {'min': v.min(), 'max': v.max(), 'mean': v.mean()}
    cyan_pixels = np.sum(cyan_mask > 0)
    
    save_to_csv(image_path, image.shape, h_stats, s_stats, v_stats, cyan_pixels, lower_cyan, upper_cyan)
    
    print("Обработка завершена успешно")
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nПрограмма прервана")
        sys.exit(0)
    except Exception as e:
        print(f"Критическая ошибка: {e}")
        sys.exit(1)