import sys
sys.path.append('e:\\Work\\University\\PR\\project1')
import numpy as np
import json
from sklearn.model_selection import train_test_split,cross_val_score
from BaseFunc import extract_features, read_wfdb
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
X=[]
file_path="E:\\Work\\University\\PR\\project1\\base-reg\\out.json"
def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()
def runModels(X,Y,name:str):
    print(name)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
    models=[SVR(kernel="linear"),SVR(kernel="poly"),SVR(kernel="rbf"),SVR(kernel="sigmoid")]
    for i in models:
        print(i.kernel)
        # آموزش مدل SVR
        model = i  # می‌تونی پارامترها رو تغییر بدی، مثل C=1.0 یا gamma='scale'
        # کراس ولیدیشن (5-Fold) روی داده train
        scores_mse = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
        scores_r2 = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        # محاسبه معیارها برای کراس ولیدیشن
        print('معیارهای Cross-Validation (روی داده train):')
        print('Mean MSE:', -scores_mse.mean())
        print('Std MSE:', scores_mse.std())
        print('Mean RMSE:', np.sqrt(-scores_mse.mean()))
        print('Mean R2:', scores_r2.mean())
        print('Std R2:', scores_r2.std())

        # آموزش مدل روی کل داده train
        model.fit(X_train, y_train)

        # پیش‌بینی و ارزیابی روی test
        y_test_pred = model.predict(X_test)
        print('\nمعیارهای Test:')
        mse_test = mean_squared_error(y_test, y_test_pred)
        print('MSE:', mse_test)
        print('RMSE:', np.sqrt(mse_test))
        print('MAE:', mean_absolute_error(y_test, y_test_pred))
        print('R2:', r2_score(y_test, y_test_pred))
        print('Explained Variance:', explained_variance_score(y_test, y_test_pred))
    

Y_VHI=[]
Y_RSI=[]
with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
for _,i in enumerate(range(1,209)):
    fs, signal= read_wfdb(f"E:/Work/University/PR/datas/voice-icar-federico-ii-database-1.0.0/voice{i:03d}")
    mel= extract_features(signal,fs)
    X.append(mel)
    metaData = next((item for item in data if item.get('ID') == f"voice{i:03d}"), None)
    Y_RSI.append( metaData["RSI"])
    Y_VHI.append(metaData["VHI"])
X=np.array(X)
Y_VHI=np.array(Y_VHI)
Y_RSI=np.array(Y_RSI)

runModels(X,Y_RSI,"RSI")
runModels(X,Y_VHI,"VHI")