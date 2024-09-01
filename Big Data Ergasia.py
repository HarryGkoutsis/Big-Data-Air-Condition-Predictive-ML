# ----------------------------------------




# Βημα 1 Εισαγωγη





import pandas as pd

# Εγκατάσταση της βιβλιοθήκης xlrd
!pip install xlrd

# Φορτώστε το σύνολο δεδομένων
file_path = '/Users/gkoutsischaralampos/Desktop/Ε Ρ Γ Α Σ Ι Ε Σ/Big Data Εργασια/Energy Consumption Time Series Dataset/KwhConsumptionBlower78_1_new.xls'
data = pd.read_excel(file_path)

# Εμφανίστε τα ονόματα των στηλών
print(data.columns)

# Εξετάστε τα πρώτα δεδομένα
print(data.head())












# ----------------------------------------






# Βημα 2 Προετοιμασια δεδομενων



# Συνδυασμός των στηλών 'Date' και 'Time' σε μια ενιαία στήλη datetime
data['datetime'] = pd.to_datetime(data['Date'].astype(str) + ' ' + data['Time'].astype(str))

# Ρύθμιση της στήλης datetime ως δείκτη
data.set_index('datetime', inplace=True)

# Ελέγξτε για κενά δεδομένα
print(data.isnull().sum())

# Αντικαταστήστε ή αφαιρέστε τα κενά δεδομένα
data = data.dropna()

# Εξετάστε τη δομή των δεδομένων
print(data.info())








# ----------------------------------------





# Βημα 3 Δημιουργια γραφηματων



import matplotlib.pyplot as plt

# Οπτικοποίηση δεδομένων κατανάλωσης ενέργειας
plt.figure(figsize=(10, 6))
plt.plot(data['Energy_Consumption'], label='Κατανάλωση Ενέργειας')
plt.title('Κατανάλωση Ενέργειας με την Πάροδο του Χρόνου')
plt.xlabel('Χρόνος')
plt.ylabel('Κατανάλωση Ενέργειας (kWh)')
plt.legend()
plt.show()










# ----------------------------------------







# Βημα 4 Ανωμαλιες στην καταναλωση ενεργειας


from sklearn.ensemble import IsolationForest

# Εκπαίδευση μοντέλου Isolation Forest για τον εντοπισμό ανωμαλιών
iso = IsolationForest(contamination=0.01)
data['anomaly'] = iso.fit_predict(data[['Energy_Consumption']])

# Οπτικοποίηση ανωμαλιών
anomalies = data[data['anomaly'] == -1]
plt.figure(figsize=(10, 6))
plt.plot(data['Energy_Consumption'], label='Κατανάλωση Ενέργειας')
plt.scatter(anomalies.index, anomalies['Energy_Consumption'], color='red', label='Ανωμαλίες')
plt.title('Ανωμαλίες στην Κατανάλωση Ενέργειας')
plt.xlabel('Χρόνος')
plt.ylabel('Κατανάλωση Ενέργειας (kWh)')
plt.legend()
plt.show()











# ----------------------------------------








# Βημα 5 Προβλεψη συντηρησης


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Προετοιμασία δεδομένων για εκπαίδευση
# Αφαίρεση της στήλης datetime από τα χαρακτηριστικά
X = data.drop(['Energy_Consumption', 'Date', 'Time'], axis=1)
y = data['Energy_Consumption']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Εκπαίδευση μοντέλου Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Πρόβλεψη και αξιολόγηση
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Οπτικοποίηση των προβλέψεων
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.title('Πρόβλεψη Κατανάλωσης Ενέργειας')
plt.xlabel('Χρόνος')
plt.ylabel('Κατανάλωση Ενέργειας (kWh)')
plt.legend()
plt.show()











# ----------------------------------------
