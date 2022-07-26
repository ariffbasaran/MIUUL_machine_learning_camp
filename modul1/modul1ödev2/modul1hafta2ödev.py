import pandas as pd
#GÖREV 1 Seaborn kütüphanesi içerisinden Titanic veri setini tanımlayınız.
import seaborn as sns
df = sns.load_dataset("titanic")
df.columns
#GÖREV 2 Yukarıda tanımlanan Titanic veri setindeki kadın ve erkek yolcuların sayısını bulunuz.

df["sex"].value_counts()

#GÖREV 3  Her bir sutuna ait unique değerlerin sayısını bulunuz.

   print(df.nunique())

#GÖREV 4 pclass değişkeninin unique değerleri bulunuz.

df["pclass"].value_counts()

#GÖREV 5 pclass ve parch değişkenlerinin unique değerlerinin sayısını bulunuz.

df[["pclass","parch"]].nunique()

#GÖREV 6  embarked değişkeninin tipini kontrol ediniz. Tipini category olarak değiştiriniz. Tekrar tipini kontrol ediniz.

print(df["embarked"].dtype)
df["embarked"] = df["embarked"].astype("category")

#GÖREV 7 embarked değeri C olanların tüm bilgelerini gösteriniz.

df[df["embarked"] == "C"]

#GÖREV 8 embarked değeri S olmayanların tüm bilgelerini gösteriniz.

df[df["embarked"] != "S"]

#GÖREV 9 Yaşı 30 dan küçük ve kadın olan yolcuların tüm bilgilerini gösteriniz.

df.loc[(df["age"] < 30) & (df["sex"] == "female")]

#GÖREV 10 Fare'i 500'den büyük veya yaşı 70 den büyük yolcuların bilgilerini gösteriniz.

df.loc[(df["fare"] > 500) | (df["age"] > 70)

#GÖREV 11 Her bir değişkendeki boş değerlerin toplamını bulunuz.

df.isnull().sum()

#GÖREV 12 who değişkenini dataframe'den düşürün.

df.drop("who", axis = 1)

#GÖREV 13 deck değikenindeki boş değerleri deck değişkenin en çok tekrar eden değeri (mode) ile doldurunuz.

mod = df["deck"].mode()[0]
df["deck"].fillna(mod , inplace= True)
df["deck"].isnull().sum()


#GÖREV 14 Age değişkenindeki boş değerleri age değişkenin medyanı ile doldurun.

median = df["age"].mode()
df["age"].fillna(median, inplace= True)
df["age"].isnull().sum()

#GÖREV 15 Survived değişkeninin Pclass ve Cinsiyet değişkenleri kırılımınında sum, count, mean değerlerini bulunuz.

df.groupby(["pclass","sex"]).agg({"survived": ["sum","count","mean"]})

#GÖREV 16   30 yaşın altında olanlar 1, 30'a eşit ve üstünde olanlara 0 vericek bir fonksiyon yazınız.
# Yazdığınız fonksiyonu kullanarak titanik veri setinde age_flag adında bir değişken oluşturunuz oluşturunuz. (apply ve lambda yapılarını kullanınız)

def age30(age):
   if age < 30:
      return 1
   else :
      return 0

df["age_flag"] = df["age"].apply(lambda x : age30(x))

#GÖREV 17 Seaborn kütüphanesi içerisinden Tips veri setini tanımlayınız.

import seaborn as sns
df_1 = sns.load_dataset("tips")
df_1.head()

#GÖREV 18 Time değişkeninin kategorilerine (Dinner, Lunch) göre total_bill  değerlerinin toplamını, min, max ve ortalamasını bulunuz.

df_1.groupby(["time"]).agg({"total_bill": ["min","max"]})

#GÖREV 19 Günlere ve time göre total_bill değerlerinin toplamını, min, max ve ortalamasını bulunuz.

df_1.groupby(["day","time"]).agg({"total_bill": ["sum","min","max","mean"]})

#GÖREV 20 Lunch zamanına ve kadın müşterilere ait total_bill ve tip  değerlerinin day'e göre toplamını, min, max ve ortalamasını bulunuz.

df_1[(df_1["time"] == "Lunch") & (df_1["sex"] == "Female")].groupby("day").agg({"total_bill": ["sum","min","max","mean"],
                                                                           "tip":  ["sum","min","max","mean"]})
#GÖREV 21 Size'i 3'ten küçük, total_bill'i 10'dan büyük olan siparişlerin ortalaması nedir?

df_1[(df_1["size"] < 3) & (df_1["total_bill"] >10 )]["total_bill"].mean()

#GÖREV 22 total_bill_tip_sum adında yeni bir değişken oluşturun. Her bir müşterinin ödediği totalbill ve tip in toplamını versin.

df_1["total_bill_tip_sum"] =  df_1["total_bill"] + df_1["tip"]
df_1.head()

# GÖREV 23 Total_bill değişkeninin kadın ve erkek için ayrı ayrı ortalamasını bulun.
# Bulduğunuz ortalamaların altında olanlara 0, üstünde ve eşit olanlara bir verildiği yeni bir total_bill_flag değişkeni oluşturun.
# Dikkat !! Female olanlar için kadınlar için bulunan ortalama dikkate alıncak, male için ise erkekler için bulunan ortalama.
# parametre olarak cinsiyet ve total_bill alan bir fonksiyon yazarak başlayın. (If-else koşulları içerecek)

f_avg = df_1[df_1["sex"]=="Female"]["total_bill"].mean()
m_avg = df_1[df_1["sex"]=="Male"]["total_bill"].mean()
def fonk(sex, total_bill):
   if sex == "male":
      if total_bill < m_avg:
         return 0
      else :
         return 1
   else :
      if total_bill < f_avg:
         return 0
      else :
         return 1

df_1["total_bill_flag"] = df_1[["sex","total_bill"]].apply(lambda x: fonk(x["sex"],x["total_bill"]),axis=1)
df_1.head()

#GÖREV 24 total_bill_flag değişkenini kullanarak cinsiyetlere göre ortalamanın altında ve üstünde olanların sayısını gözlemleyin.

df_1.groupby(["sex","total_bill_flag"]).agg({"total_bill_flag": ["count"]})

#GÖREV 25 total_bill_tip_sum değişkenine göre büyükten küçüğe sıralayınız ve ilk 30 kişiyi yeni bir dataframe'e atayınız.

new_df = df_1.sort_values("total_bill_tip_sum",ascending = False)[:30]
new_df.head()