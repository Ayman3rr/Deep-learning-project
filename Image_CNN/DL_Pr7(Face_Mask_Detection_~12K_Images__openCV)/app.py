import cv2
import numpy as np
from tensorflow.keras.models import load_model

# تحميل النموذج
model = load_model('C:/Users/moham/Desktop/DL_Pr7(Face_Mask_Detection_~12K_Images__openCV)/FaceMask.h5')

# أسماء الفئات
class_names = ['WithMask', 'WithoutMask']

# فتح الكاميرا
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # تحويل الصورة إلى اللون الرمادي
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # الكشف عن الوجه
    faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(gray, 1.1, 4)
    
    for (x, y, w, h) in faces:
        # رسم مستطيل حول الوجه
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # استخراج الوجه
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (224, 224))  # حجم الصورة حسب النموذج
        
        # تحضير الصورة للتنبؤ
        face_input = face / 255.0
        face_input = np.expand_dims(face_input, axis=0)
        
        # التنبؤ بالقناع
        prediction = model.predict(face_input)
        class_index = np.argmax(prediction[0])
        class_name = class_names[class_index]
        
        # عرض النتيجة على الإطار
        cv2.putText(frame, class_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # عرض الإطار
    cv2.imshow('Face Mask Detection', frame)
    
    # إنهاء البرنامج عند الضغط على مفتاح 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# إغلاق الكاميرا والنوافذ
cap.release()
cv2.destroyAllWindows()
