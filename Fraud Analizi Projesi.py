#RANDOM FOREST İLE

import pandas as pd

dosya_yolu = pd.read_csv('creditcard.csv')  
df = pd.read_csv(dosya_yolu)
print(df.shape)

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE 

# 1. CHECKPOINT: Orijinal veriyi koruyalım
# df zaten hafızada yüklü, onu ellemiyoruz.
df_calisma = df.copy() 

print("Kopya oluşturuldu(df_calisma)")

# 2. İşlemleri 'df_calisma' üzerinde yapıyoruz
scaler = StandardScaler()
# Amount'u ölçekleyip yeni sütuna atıyoruz
df_calisma['normAmount'] = scaler.fit_transform(df_calisma['Amount'].values.reshape(-1, 1))

# Gereksizleri atıyoruz
df_calisma = df_calisma.drop(['Time', 'Amount'], axis=1)

# 3. x ve y ayırıyoruz
x = df_calisma.drop('Class', axis=1)
y = df_calisma['Class']

# 4. Böl (%30 Test)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y)

# 5. SMOTE ile Dengele (Eğitim setindeki hırsızları çoğaltıyoruz)
sm = SMOTE(random_state=42)
x_train_res, y_train_res = sm.fit_resample(x_train, y_train)

print(f"Eskiden {y_train.value_counts()[1]} tane hırsız vardı.")
print(f"Şimdi {y_train_res.value_counts()[1]} tane hırsız var (Eşitlendi).")


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt


# Modeli kuruyoruz (100 tane karar ağacı kullanacak)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# **Modeli SMOTE'lu (dengeli) veriyle eğitiyoruz
rf_model.fit(x_train_res, y_train_res)

# Ama testi gerçek (dengesiz) veriyle yapıyoruz ki gerçek performansını görelim
y_pred = rf_model.predict(x_test)

print("\nConfusion Matrix (Hata Matrisi):")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\nSınıflandırma Raporu:")
print(classification_report(y_test, y_pred))

# Görselleştirelim ki mülakatta gösterirsin
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Fraud Tespit Sonuçları (0: Normal, 1: Fraud)')
plt.ylabel('Gerçek Durum')
plt.xlabel('Modelin Tahmini')
plt.show()

from sklearn.metrics import confusion_matrix, recall_score

# 1. Modelin olasılık tahminlerini alalım
olasiliklar = rf_model.predict_proba(x_test)[:, 1]

# 2. %30'dan (0.3) fazla şüpheleniyorsan "Hırsız" (1) de
yeni_tahminler = (olasiliklar > 0.22).astype(int)

# 3. Sonuçları yazdır
print("--- EŞİK DEĞERİ 0.22 SONUÇLARI ---")
cm = confusion_matrix(y_test, yeni_tahminler)
print(cm)

kacan_hirsiz = cm[1][0]
yakalanan = cm[1][1]
yeni_basari = recall_score(y_test, yeni_tahminler)

print(f"\nKaçırdığımız Hırsız Sayısı: {kacan_hirsiz} (Eskiden 32 idi!)")
print(f"Yakalanan Hırsız Sayısı:    {yakalanan}")
print(f"Yeni Başarı Oranı (Recall): %{yeni_basari*100:.2f}")

import joblib

# Modeli bilgisayarına dosya olarak kaydediyoruz
joblib.dump(rf_model, 'fraud_modeli_v1.pkl')

############################################################################3333
#ISOLATION FOREST İLE
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, recall_score
import numpy as np

# 1. Isolation Forest'ı Eğitiyoruz
# contamination=0.002 (Binde 2 hırsız var varsayımıyla)
iso_model = IsolationForest(n_estimators=100, contamination=0.002, random_state=42, n_jobs=-1)
iso_model.fit(x_train) # SMOTE'suz veriyle eğitiyoruz.**

# 2. Tahmin 
y_pred_iso_raw = iso_model.predict(x_test)

# 3. Sonuçları 0 ve 1'e çeviriyoruz (-1 hırsız demektir, onu 1 yapıyoruz)
y_pred_iso = np.where(y_pred_iso_raw == -1, 1, 0)

# 4. Raporla
cm = confusion_matrix(y_test, y_pred_iso)
print("\nConfusion Matrix (Sadece IF):")
print(cm)

kacan = cm[1][0]
yakalanan_hirsiz = cm[1][1]
rahatsiz_edilen = cm[0][1]  # Normal olduğu halde Fraud denilenler (Yanlış Alarm)
basari = recall_score(y_test, y_pred_iso)

print(f"\nKaçan Hırsız: {kacan}")
print(f"Yakalanan Hırsız: {yakalanan}")
print(f"Başarı (Recall): %{basari*100:.2f}")
print(f"Rahatsız Edilen Masum Müşteri: {rahatsiz_edilen}")
print(f"\nSonuç: Her 1 hırsız yakalamak için yaklaşık {int(rahatsiz_edilen/yakalanan_hirsiz)} masum müşteriyi yakalandı.")
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Temiz (0)', 'Hırsız (1)'])
disp.plot(cmap='Oranges', values_format='d')
plt.title('Isolation Forest Sonuç Tablosu')
plt.show()

####################################################################################################
# Hibrit Model(Random + Isolation Forest)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import confusion_matrix, recall_score
from imblearn.over_sampling import SMOTE

# 1. Veriyi Yüklüyoruz
dosya_yolu = pd.read_csv('creditcard.csv')  
df = pd.read_csv(dosya_yolu)

# 2. Temizliyoruz
scaler = StandardScaler()
df['normAmount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df = df.drop(['Time', 'Amount'], axis=1)

x = df.drop('Class', axis=1)
y = df['Class']

# 3. Bölme (%30 Test) 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y)

# 4. Random Forest Hazırlığı
sm = SMOTE(random_state=42)
x_train_res, y_train_res = sm.fit_resample(x_train, y_train)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(x_train_res, y_train_res)

# Formül için olasılık alıyoruz (predict_proba)
y_pred_rf_prob = rf.predict_proba(x_test)[:, 1]
y_pred_rf_karar = (y_pred_rf_prob > 0.22).astype(int)

# 5. Isolation Forest Hazırlığı
iso = IsolationForest(n_estimators=100, contamination=0.002, random_state=42, n_jobs=-1)
iso.fit(x_train)

y_pred_diger_model_prob = np.where(iso.predict(x_test) == -1, 1, 0)

# 6. HİBRİT KARAR (SENİN FORMÜL)
print("Modeller birleştiriliyor...")

# 1. Önce RF'in ham olasılıklarını alıyoruz
y_pred_rf_prob = rf.predict_proba(x_test)[:, 1]

# 2. Eşik Değer (0.22)
y_pred_rf_karar = (y_pred_rf_prob > 0.22).astype(int)

# 3. Formülde bu kararı kullanıyoruz (Ağırlıklar: %80 RF, %20 IF)
y_pred_hybrid_score = (y_pred_rf_karar * 0.80) + (y_pred_diger_model_prob * 0.20)

# Sonuç hesapla
y_pred_final = (y_pred_hybrid_score > 0.5).astype(int)

# 7. SONUÇ
cm = confusion_matrix(y_test, y_pred_final)
recall = recall_score(y_test, y_pred_final)
rahatsiz_edilen_masum = cm[0][1]


print("\n--- (Ağırlıklı Hibrit) ---")
print(f"Kaçan Hırsız Sayısı: {cm[1][0]}")
print(f"Yakalanan Hırsız:    {cm[1][1]}")
print(f"Rahatsız Edilen Masum Müşteri: {rahatsiz_edilen_masum}")
print(f"TOPLAM BAŞARI (Recall): %{recall*100:.2f}")

from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Bu araç, renk skalası bozulsa bile sayıları içine net yazar.
# 'values_format' kısmını 'd' yaptık ki bilimsel sayı (1e3 gibi) göstermesin.
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Temiz (0)', 'Hırsız (1)'])
plt.figure(figsize=(8, 6))
disp.plot(cmap='Blues', values_format='d')

plt.title('Hırsız Yakalama Sonuçları')
plt.show()
#######################################################################################################
