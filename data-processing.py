# 📌 1. Gerekli Kütüphaneleri İçe Aktarma
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

#  2. Veri Setini Yükleme
df = pd.read_csv("titanic.csv")

#  3. İlk İncelemeler
print(df.head())                # İlk 5 satırı görüntüle
print(df.info())                # Veri tipi ve eksik değer bilgisi
print(df.isnull().sum())        # Hangi sütunda kaç eksik veri var?

#  4. Kullanılacak Sütunları Seçme
df = df[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]

#  5. Eksik Verileri Doldurma
df['Age'].fillna(df['Age'].mean(), inplace=True)                        # Age: Ortalama ile
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)          # Embarked: En sık görülen değer ile

#  6. Kategorik Değişkenleri Sayısal Formata Çevirme
df['Sex'] = LabelEncoder().fit_transform(df['Sex'])
df['Embarked'] = LabelEncoder().fit_transform(df['Embarked'])

#  7. Giriş (X) ve Çıkış (y) Değişkenlerini Ayırma
X = df.drop('Survived', axis=1)
y = df['Survived']

#  8. Eğitim ve Test Verisi Olarak Ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  9. Model Oluşturma ve Eğitme
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

#  10. Modeli Test Etme ve Değerlendirme
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

#  11. Embarked Dağılımını Görselleştirme
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='Embarked', palette='pastel')
plt.title('Yolcuların Bindiği Liman Dağılımı', fontsize=14)
plt.xlabel('Liman (Embarked)')
plt.ylabel('Yolcu Sayısı')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

#  12. Korelasyon Isı Haritası
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Veri Değişkenleri Korelasyon Isı Haritası')
plt.show()