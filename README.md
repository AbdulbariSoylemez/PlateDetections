
# Araç Plaka Tespiti ve İnterpolasyon

Bu Python projesi, bir video kaydındaki araçları tespit edip, plakalarını okuyarak eksik verileri interpolasyon yöntemiyle doldurmayı amaçlamaktadır. Aynı zamanda elde edilen sonuçları görsel olarak izlenebilen bir çıkış video dosyasına aktarır.

## Proje Amaçları

1. **Araç Tespiti:** YOLO (You Only Look Once) algoritması kullanılarak videodaki araçlar tespit edilir.

2. **Plaka Tespiti:** Ayrı bir YOLO modeli ile araçların üzerindeki plakalar tespit edilir.

3. **Veri İnterpolasyonu:** Araçların ve plakaların tespit edildiği her karede, eksik veri olup olmadığı kontrol edilir. Eksik veriler varsa, bu karelerdeki bilgiler interpolasyon yöntemiyle doldurulur.

4. **Sonuçların Kaydedilmesi:** Elde edilen sonuçlar bir CSV dosyasına kaydedilir. Bu dosyada her karede tespit edilen araç ve plakaların bilgileri bulunur.

5. **Görsel İzleme:** Proje, tespit edilen araçları ve plakaları içeren bir çıkış video dosyası oluşturur. Bu video, tespitlerin görsel olarak nasıl yapıldığını izlemek için kullanılabilir.

## Kullanılan Kütüphaneler

- **Ultralytics YOLO:** YOLO modeli için Ultralytics kütüphanesinin kullanılması.
- **OpenCV:** Görüntü işleme ve video işleme için kullanılır.
- **NumPy:** Bilimsel hesaplamalar ve veri manipülasyonu için kullanılır.
- **Sort:** Çoklu nesne takibi için kullanılan bir algoritma.
- **EasyOCR:** Optik karakter tanıma (OCR) için kullanılır.
- **Scipy.interpolate:** Veri interpolasyonu için kullanılır.

