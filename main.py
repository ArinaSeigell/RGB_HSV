import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1
image = cv2.imread('GettyImages-1324644587.jpg') 
if image is None:
    print("Ошибка: Не удалось загрузить изображение!")
    exit()  # Завершаем программу при ошибке загрузки
else:
    print("Изображение успешно загружено")

# 2
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 3
h, s, v = cv2.split(hsv_image)

# 4
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Оригинальное изображение')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(h, cmap='gray')
plt.title('Канал H (Hue)')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(s, cmap='gray')
plt.title('Канал S (Saturation)')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(v, cmap='gray')
plt.title('Канал V (Value)')
plt.axis('off')

# 5
lower_cyan = np.array([85, 50, 50])    
upper_cyan = np.array([100, 255, 255]) 

cyan_mask = cv2.inRange(hsv_image, lower_cyan, upper_cyan)

cyan_only = cv2.bitwise_and(image, image, mask=cyan_mask)

plt.subplot(2, 3, 5)
plt.imshow(cyan_mask, cmap='gray')
plt.title('Маска голубого цвета')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(cv2.cvtColor(cyan_only, cv2.COLOR_BGR2RGB))
plt.title('Только голубые объекты')
plt.axis('off')

plt.tight_layout()
plt.show()
