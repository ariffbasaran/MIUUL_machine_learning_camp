
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
#!pip install missingno
import missingno as msno
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.simplefilter(action="ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)
data = pd.read_csv("diabetes.csv")

##############################################################
#             GÖREV 1: Keşifçi Veri Analizi                  #
##############################################################

#ADIM 1 Genel resmi inceleyiniz.

    print("##### Shape #####")
    print(data.shape)
    print("\n########### Types ###########")
    print(data.dtypes)
    print("\n################################ Head ################################")
    print(data.head())
    print("\n################################ Tail ################################")
    print(data.tail())
    print("\n######### NA #########")
    print(data.isnull().sum())
    print("\n################################ Quantiles ################################")
    print(data.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)



#ADIM 2 Numerik ve kategorik değişkenleri yakalayınız.

def grab_col_names(dataframe, cat_th=10, car_th=20):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(data)

#ADIM 3 Numerik ve kategorik değişkenlerin analizini yapınız.

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


for col in cat_cols:
    cat_summary(data, col, plot=True)

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in num_cols:
    num_summary(data, col, plot=True)

#ADIM 4 Hedef değişken analizi yapınız.
# (Kategorik değişkenlere göre hedef değişkenin ortalaması, hedef değişkene göre numerik değişkenlerin ortalaması)

def hedef_degiskene_göre_num (dataframe, target, num_col):
    print(dataframe.groupby(target).agg({num_col: ["mean"]}), end="\n\n")

for col in num_cols:
    hedef_degiskene_göre_num (data, ["Outcome"], col)

def hedef_degiskene_göre_cat(dataframe, target, cat_col):
    print(pd.DataFrame({"Target_Mean": dataframe.groupby(cat_col)[target].mean(),
                        "Value_Count": dataframe.groupby(cat_col)[target].count()}))


for col in cat_cols:
    hedef_degiskene_göre_cat(data, "Outcome", col)

#ADIM 5 Aykırı Değer Analizi
def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

 check_outlier(data,"Insulin")

def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index

grab_outliers(data, "BMI")

#ADIM 6 Eksik gözlem analizi yapınız.

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()

    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)

    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns

    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")


missing_values_table(data, na_name= True)

#ADIM 7 Korelasyon analizi yapınız.

def high_correlated_cols(dataframe, plot=False, corr_th=0.90):
    corr = dataframe.corr()
    corr_matrix = corr.abs()
    upper_triangle_matrix = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={"figure.figsize": (10, 10)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
    return drop_list


high_cor_cols = high_correlated_cols(data, plot= True)

##############################################################
#              GÖREV 2: Feature Engineering                  #
##############################################################

#ADIM 1
#0 içeren değişkenleri bulup anlamlı olup olmadığına bakacağız anlamsızlar boş olanlar olacak.
# Gerekli hesaplamaları yapabilmek için yakaladığımız boş değerlere NaN atayacağız.
data.min()
meaningles = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI" ]

def check_zeros(dataframe, zero_cols):
    n_zero = dataframe[zero_cols].isin([0]).sum().sort_values(ascending=False)
    ratio = (dataframe[zero_cols].isin([0]).sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_zero, np.round(ratio, 2)], axis=1, keys=['n_zero', 'ratio'])
    print(missing_df, end="\n")


check_zeros(data, meaningles)

for col in meaningles:
    data[col] = data[col].replace({'0':np.nan, 0:np.nan})

missing_values_table(data)
missing_vs_target(data, "Outcome", meaningles)

#Boş değerleri KNNI yöntemi ile doldurduk.

from sklearn.impute import KNNImputer
Imputer = KNNImputer(n_neighbors=5)
data = pd.DataFrame(Imputer.fit_transform(data), columns=data.columns)

#Şimdi ise aykırı değerlerden kurtulmamız lazım.

def replace_with_thresholds(dataframe, col_name):
    low, up = outlier_thresholds(dataframe, col_name)

    dataframe.loc[dataframe[col_name] > up, col_name] = up
    dataframe.loc[dataframe[col_name] < low, col_name] = low
for col in num_cols:
    replace_with_thresholds(data, col)
for col in num_cols:
    print(col, check_outlier(data, col))

#Çok değişkenli aykırılık analizi yapıyoruz.

clf = LocalOutlierFactor(n_neighbors=20)
clf.fit_predict(data)
df_scores = clf.negative_outlier_factor_

df_scores[0:5]
np.sort(df_scores)[0:5]

#Verilerimizi görselleştiriyotuz ve bir eşlik değer belirliyotuz ani sıçramaya göre.
scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True, xlim=[0, 20], style='.-')
plt.show()

th = np.sort(df_scores)[4] #Eşlik değerimiz 4. değer

#4'ün altında kalan aykırı verileri inceliyoruz.
data[df_scores < th]
data[df_scores < th].shape

data.describe([0.01, 0.05, 0.75, 0.9, 0.99]).T

#Aykırı verileri siliyoruz.
data[df_scores < th].index
data.drop(data[df_scores < th].index, axis=0, inplace=True)

#ADIM 2 Yeni değişken üretiyoruz.

data['HighBloodPressure'] = data['BloodPressure'].apply(lambda x: 1 if x >= 90 else 0)
data.groupby('HighBloodPressure').agg({'Outcome': 'mean'})

#Yeni değişken anlamlı mı? diye oran testi yapıyoruz.
from statsmodels.stats.proportion import proportions_ztest
test_stats, pvalue = proportions_ztest(count=[data.loc[data["HighBloodPressure"] == 1, "Outcome"].sum(),
                                              data.loc[data["HighBloodPressure"] == 0, "Outcome"].sum()],
                                       nobs=[data.loc[data["HighBloodPressure"] == 1, "Outcome"].shape[0],
                                             data.loc[data["HighBloodPressure"] == 0, "Outcome"].shape[0]])


print('Test stat= %.4f, p-value %.4f' % (test_stats, pvalue))
# pvalue, < 0.05 ise h0 hipotezi reddedilir, h1 kabul edilir.
# h0: Kan basıncının yüksek olması ile diyabet arasında anlamlı fark yoktur
# h1: Kan basıncının yüksek olması ile diyabet arasında anlamlı fark vardır.
# Kan basıncının yüksek olması ile diyabet arasında anlamlı bir fark vardır.



data. loc[(data.BMI <= 18) & (data.Glucose <= 100), "BMI_GLUCOSE"] = "LOW_BMI_LOW_GLUCOSE"
data.loc[(data.BMI <= 25) & (data.Glucose <= 100), "BMI_GLUCOSE"] = "NORMAL_BMI_LOW_GLUCOSE"
data.loc[(data.BMI > 30)  &  (data.Glucose <= 100), "BMI_GLUCOSE"] = "HIGH_BMI_LOW_GLUCOSE"

data. loc[(data.BMI <= 18) & (data.Glucose > 100) & (data.Glucose <= 140), "BMI_GLUCOSE"] = "LOW_BMI_NORMAL_GLUCOSE"
data.loc[(data.BMI <= 25) & (data.BMI > 18) & (data.Glucose <= 100), "BMI_GLUCOSE"] = "NORMAL_BMI_NORMAL_GLUCOSE"
data.loc[(data.BMI > 30)  &  (data.Glucose > 100) & (data.Glucose <= 140), "BMI_GLUCOSE"] = "HIGH_BMI_NORMAL_GLUCOSE"

data. loc[(data.BMI <= 18) & (data.Glucose > 140), "BMI_GLUCOSE"] = "LOW_BMI_HIGH_GLUCOSE"
data.loc[(data.BMI <= 25) & (data.Glucose > 140), "BMI_GLUCOSE"] = "NORMAL_BMI_HIGH_GLUCOSE"
data.loc[(data.BMI > 30)  &  (data.Glucose > 140), "BMI_GLUCOSE"] = "HIGH_BMI_HIGH_GLUCOSE"

data.groupby("BMI_GLUCOSE")["Outcome"].mean()

data.columns

# Kolonların büyültülmesi
data.columns = [col.upper() for col in data.columns]

data.head()
data.shape

#ADIM 3 Encoding
cat_cols, num_cols, cat_but_car = grab_col_names(data)

# LABEL ENCODING
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


binary_cols = [col for col in data.columns if data[col].dtypes == "O" and data[col].nunique() == 2]
binary_cols

for col in binary_cols:
    data = label_encoder(data, col)

data.head()

# One-Hot Encoding İşlemi
# cat_cols listesinin güncelleme işlemi
cat_cols = [col for col in cat_cols if col not in binary_cols and col not in ["OUTCOME"]]
cat_cols


def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


data = one_hot_encoder(data, cat_cols, drop_first=True)

data.shape
data.head()

#ADIM 4 Standartlaştırm

num_cols

scaler = StandardScaler()
data[num_cols] = scaler.fit_transform(data[num_cols])

data.head()
data.shape

#ADIM 5 Model

y = data["OUTCOME"]
X = data.drop("OUTCOME", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 2)}")
print(f"Recall: {round(recall_score(y_pred, y_test), 3)}")
print(f"Precision: {round(precision_score(y_pred, y_test), 2)}")
print(f"F1: {round(f1_score(y_pred, y_test), 2)}")
print(f"Auc: {round(roc_auc_score(y_pred, y_test), 2)}")


# Accuracy: 0.79
# Recall: 0.711
# Precision: 0.67
# F1: 0.69
# Auc: 0.77

# Base Model
# Accuracy: 0.77
# Recall: 0.706
# Precision: 0.59
# F1: 0.64
# Auc: 0.75

#Feature Importance
def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    print(feature_imp.sort_values("Value", ascending=False))
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(rf_model, X)
